// Minimal G1: FP64, one CUDA thread per particle, full all-pairs forces.
// Build: nvcc -O2 -std=c++17 --fmad=false pasta_g1_simple.cu -o pasta_g1
// Run:   ./pasta_g1 [N|M1-checkpoint.txt] [cutoff] [repeats]
// CPU only: g++ -x c++ -O2 -std=c++17 pasta_g1_simple.cu -o pasta_cpu

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HD __host__ __device__
#else
#define HD
#endif

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct Box {
    int n;
    double L, cutoff = 6.0;
    std::vector<double> xyz; // SoA: x[0..n), y[n..2n), z[2n..3n).
    std::vector<int> species; // 0 = neutron, 1 = proton.
};

Box read_box(const std::string& source) {
    Box b;
    double density = 0.05;
    if (!source.empty() && source.find_first_not_of("0123456789") == std::string::npos) {
        b.n = std::stoi(source);
        require(b.n >= 2 && b.n <= 100000, "Require 2 <= N <= 100000");
        b.L = std::cbrt(b.n / density);
        b.xyz.resize(3 * b.n);
        b.species.assign(b.n, 0);
        std::mt19937_64 rng(20261017);
        for (double& x : b.xyz)
            x = static_cast<double>(rng() >> 11) * 0x1.0p-53 * b.L;
        std::fill_n(b.species.begin(), static_cast<int>(0.3 * b.n), 1);
        std::shuffle(b.species.begin(), b.species.end(), rng);
    } else {
        // Only positions/species are needed; read and discard velocities.
        std::ifstream file(source);
        file.imbue(std::locale::classic());
        std::string magic, version;
        int protons;
        double saved_L, yp, dt, time, velocity;
        unsigned long long step, seed;
        std::mt19937_64 saved_rng;
        file >> magic >> b.n >> protons;
        require(file && magic == "M1_TEXT_V1", "Expected M1_TEXT_V1 checkpoint");
        require(b.n >= 2 && b.n <= 100000, "Invalid checkpoint N");
        file >> density >> saved_L >> yp >> b.cutoff >> dt >> time
             >> step >> seed >> std::quoted(version) >> saved_rng;
        require(file && std::isfinite(density) && density > 0, "Invalid checkpoint header");
        b.L = std::cbrt(b.n / density);
        require(std::isfinite(saved_L) && saved_L > 0 &&
                std::abs(b.L - saved_L) <= 1e-14 * saved_L, "Inconsistent checkpoint box");
        b.xyz.resize(3 * b.n);
        b.species.resize(b.n);
        for (double& x : b.xyz) file >> x;
        for (int k = 0; k < 3 * b.n; ++k) file >> velocity;
        for (int& q : b.species) file >> q;
        require(static_cast<bool>(file), "Truncated checkpoint");
        require(protons == std::count(b.species.begin(), b.species.end(), 1),
                "Checkpoint proton count mismatch");
    }
    require(std::isfinite(b.L) && b.L > 0, "Invalid box length");
    for (double x : b.xyz)
        require(std::isfinite(x) && x >= 0 && x < b.L, "Position outside [0,L)");
    for (int q : b.species) require(q == 0 || q == 1, "Invalid species");
    return b;
}

struct Pair { double B, K, cutoff_force; };
struct Model {
    double width = 1.25, lambda = 10.0, cutoff;
    Pair pairs[4]; // nn, np, pn, pp.
};

HD double minimum_image(double d, double L) {
    return d - L * nearbyint(d / L);
}

// V(r) = 110 exp(-r^2/width) + B exp(-r^2/(2*width))
//        + K exp(-r/lambda)/r. Return -V'(r)/r so F_ij = factor*(r_i-r_j).
HD double raw_factor(double r2, const Pair& p, const Model& m) {
    const double r = sqrt(r2);
    return (220.0 / m.width) * exp(-r2 / m.width)
         + (p.B / m.width) * exp(-r2 / (2.0 * m.width))
         + p.K * exp(-r / m.lambda) * (1.0 / (m.lambda*r2) + 1.0 / (r2*r));
}

Model make_model(double cutoff, double L) {
    require(std::isfinite(cutoff) && cutoff > 0 && cutoff <= L/2,
            "Require 0 < cutoff <= L/2; try N=200, cutoff=6");
    Model m;
    m.cutoff = cutoff;
    for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) {
        Pair p{a == b ? -2.0 : -50.0, a == 1 && b == 1 ? 1.44 : 0.0, 0.0};
        p.cutoff_force = cutoff * raw_factor(cutoff*cutoff, p, m);
        m.pairs[2*a+b] = p;
    }
    return m;
}

// Shared arithmetic keeps the CPU/GPU comparison small and easy to follow.
// Each invocation owns particle i's output. No atomics or shared-memory tiles.
HD void force_particle(int i, int n, double L, const double* xyz,
                       const int* species, double* force, const Model& m) {
    double fx = 0, fy = 0, fz = 0;
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        const double dx = minimum_image(xyz[i] - xyz[j], L);
        const double dy = minimum_image(xyz[n+i] - xyz[n+j], L);
        const double dz = minimum_image(xyz[2*n+i] - xyz[2*n+j], L);
        const double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= m.cutoff*m.cutoff) continue;
        if (r2 == 0) { fx = fy = fz = NAN; break; } // Reject overlaps; no softening.
        const Pair& p = m.pairs[2*species[i] + species[j]];
        // Shift radial force by F(cutoff), making it continuous at the cutoff.
        const double factor = raw_factor(r2, p, m) - p.cutoff_force/sqrt(r2);
        fx += factor*dx;
        fy += factor*dy;
        fz += factor*dz;
    }
    force[i] = fx;
    force[n+i] = fy;
    force[2*n+i] = fz;
}

#ifdef __CUDACC__
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__global__ void forces_g1(int n, double L, const double* xyz,
                          const int* species, double* force, Model model) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) force_particle(i, n, L, xyz, species, force, model);
}

void run_gpu(const Box& b, const Model& m, const std::vector<double>& cpu,
             int repeats, double cpu_ms) {
    double *d_xyz = nullptr, *d_force = nullptr;
    int* d_species = nullptr;
    const std::size_t bytes = b.xyz.size() * sizeof(double);
    check(cudaMalloc(reinterpret_cast<void**>(&d_xyz), bytes));
    check(cudaMalloc(reinterpret_cast<void**>(&d_force), bytes));
    check(cudaMalloc(reinterpret_cast<void**>(&d_species), b.n * sizeof(int)));
    check(cudaMemcpy(d_xyz, b.xyz.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_species, b.species.data(), b.n * sizeof(int), cudaMemcpyHostToDevice));
    auto launch = [&]() {
        forces_g1<<<(b.n + 255)/256, 256>>>(b.n, b.L, d_xyz, d_species, d_force, m);
        check(cudaGetLastError());
    };
    for (int k = 0; k < 3; ++k) launch(); // Warm up before timing.
    check(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    for (int k = 0; k < repeats; ++k) launch();
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float elapsed_ms;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::vector<double> gpu(cpu.size());
    check(cudaMemcpy(gpu.data(), d_force, bytes, cudaMemcpyDeviceToHost));
    int device;
    cudaDeviceProp properties;
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(d_xyz));
    check(cudaFree(d_force));
    check(cudaFree(d_species));

    double max_abs = 0, max_rel = 0;
    bool agrees = true;
    for (std::size_t k = 0; k < cpu.size(); ++k) {
        require(std::isfinite(gpu[k]), "Nonfinite GPU force");
        const double error = std::abs(cpu[k] - gpu[k]);
        max_abs = std::max(max_abs, error);
        if (cpu[k] != 0) {
            const double relative = error / std::abs(cpu[k]);
            max_rel = std::max(max_rel, relative);
            agrees = agrees && relative < 1e-12;
        } else agrees = agrees && error <= 1e-12; // Relative error undefined at zero.
    }
    std::cout << "GPU: " << properties.name << " | nvcc "
              << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '\n'
              << "GPU kernel ms/evaluation: " << elapsed_ms/repeats << '\n'
              << "Serial CPU / GPU kernel ratio (copies excluded): "
              << cpu_ms/(elapsed_ms/repeats) << '\n'
              << "Max absolute error: " << max_abs << '\n'
              << "Max component relative error: " << max_rel << '\n';
    require(agrees, "CPU/GPU comparison failed");
    std::cout << "PASS: CPU/GPU force comparison\n";
}
#endif

int main(int argc, char** argv) try {
    require(argc <= 4, "Usage: ./pasta_g1 [N|M1-checkpoint] [cutoff] [repeats]");
    const Box b = read_box(argc > 1 ? argv[1] : "200");
    const Model m = make_model(argc > 2 ? std::stod(argv[2]) : b.cutoff, b.L);
    const int repeats = argc > 3 ? std::stoi(argv[3]) : 20;
    require(repeats > 0 && repeats <= 10000, "Require 1 <= repeats <= 10000");
    std::vector<double> cpu(3*b.n);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < b.n; ++i)
        force_particle(i, b.n, b.L, b.xyz.data(), b.species.data(), cpu.data(), m);
    const double cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    for (double f : cpu) require(std::isfinite(f), "Nonfinite CPU force; check overlaps");
    std::cout << std::scientific << std::setprecision(6)
              << "N=" << b.n << " L=" << b.L << " cutoff=" << m.cutoff
              << " lambda=" << m.lambda << " | FP64, block=256\n"
              << "Serial CPU ms (one evaluation): " << cpu_ms << '\n'
              << "Force on particle 0: " << cpu[0] << ' ' << cpu[b.n]
              << ' ' << cpu[2*b.n] << '\n';
#ifdef __CUDACC__
    run_gpu(b, m, cpu, repeats, cpu_ms);
#else
    std::cout << "CPU evaluation only. Build with nvcc to run/compare the GPU kernel.\n";
#endif
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}