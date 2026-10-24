// Units: positions in fm, density in fm^-3, velocities in c, time in fm/c.
// All doubles use max_digits10 and the classic locale for lossless round-trip.

// Build: c++ -std=c++17 -O0 -g -Wall -Wextra -Wpedantic m1_box_checkpoint.cpp -o m1_box_checkpoint
// Run:   ./m1_box_checkpoint

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

void near(double actual, double expected) {
    require(std::abs(actual - expected) < 1e-12, "Geometry hand check failed");
}

double minimum_image(double displacement, double L) {
    // Handles displacements spanning multiple boxes. At a half-box tie either
    // endpoint is possible; nearbyint keeps opposite displacements antisymmetric.
    return displacement - L * std::nearbyint(displacement / L);
}

double distance(double xi, double yi, double zi,
                double xj, double yj, double zj, double L) {
    const double dx = minimum_image(xi - xj, L);
    const double dy = minimum_image(yi - yj, L);
    const double dz = minimum_image(zi - zj, L);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double wrap(double position, double L) {
    const double wrapped = position - L * std::floor(position / L);
    return (wrapped == 0.0 || wrapped >= L) ? 0.0 : wrapped;
}

struct State {
    // Inputs N and density will be used to determine L.
    double density = 100.0 / (20.0 * 20.0 * 20.0);
    double proton_fraction = 0.3;
    double cutoff = 8.0;          // Geometry-test value
    double dt = 0.5;
    double time = 0.0;
    std::uint64_t step = 0;
    std::uint64_t seed = 20261006;
    std::string code_version = "M1-v1";
    std::mt19937_64 rng;
    std::vector<double> x, y, z, vx, vy, vz;
    std::vector<int> species;

    std::size_t size() const { return x.size(); }
    double box_length() const {
        return std::cbrt(static_cast<double>(size()) / density);
    }
};

void validate(const State& s) {
    require(s.size() > 0 && s.size() <= 1000, "M1 requires 1 <= N <= 1000");
    require(std::isfinite(s.density) && s.density > 0.0, "Invalid density");
    require(std::isfinite(s.proton_fraction) && s.proton_fraction >= 0.0 &&
                s.proton_fraction <= 1.0, "Invalid proton fraction");
    const double L = s.box_length();
    require(std::isfinite(L) && L > 0.0, "Invalid box length");
    require(std::isfinite(s.cutoff) && s.cutoff > 0.0 && s.cutoff <= L / 2.0,
            "Cutoff must satisfy 0 < r_c <= L/2");
    require(std::isfinite(s.dt) && s.dt > 0.0 &&
                std::isfinite(s.time) && s.time >= 0.0, "Invalid time or dt");
    for (const auto* array : {&s.x, &s.y, &s.z, &s.vx, &s.vy, &s.vz}) {
        require(array->size() == s.size(), "State array length mismatch");
        for (double value : *array)
            require(std::isfinite(value), "Nonfinite state value");
    }
    for (const auto* array : {&s.x, &s.y, &s.z})
        for (double value : *array)
            require(value >= 0.0 && value < L, "Positions must be in [0,L)");
    require(s.species.size() == s.size(), "Species array length mismatch");
    for (int species : s.species)
        require(species == 0 || species == 1, "Unknown species label");
}

State make_test_state() {
    State s;
    constexpr std::size_t N = 100;
    for (auto* array : {&s.x, &s.y, &s.z, &s.vx, &s.vy, &s.vz})
        array->resize(N);
    s.species.resize(N);
    s.rng.seed(s.seed);
    const double L = s.box_length();
    auto uniform = [&]() { return static_cast<double>(s.rng() >> 11) * 0x1.0p-53 * L; };
    const auto proton_count = static_cast<std::size_t>(std::floor(s.proton_fraction * N));
    for (std::size_t i = 0; i < N; ++i) {
        s.x[i] = uniform(); s.y[i] = uniform(); s.z[i] = uniform();
        s.species[i] = (i < proton_count) ? 1 : 0;
        // These are test data
        s.vx[i] = 0.001 * (static_cast<double>(i) - 50.0);
        s.vy[i] = -0.5 * s.vx[i];
        s.vz[i] = 0.25 * s.vx[i];
    }
    s.vx[0] = -0.0;  // Verification for zero sign
    validate(s);
    return s;
}

void save_checkpoint(const State& s, const std::string& path) {
    validate(s);
    std::ofstream out(path);
    require(static_cast<bool>(out), "Cannot open checkpoint for writing");
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "M1_TEXT_V1\n" << s.size() << ' '
        << std::count(s.species.begin(), s.species.end(), 1) << '\n'
        << s.density << ' ' << s.box_length() << ' ' << s.proton_fraction << '\n'
        << s.cutoff << ' ' << s.dt << ' ' << s.time << ' '
        << s.step << ' ' << s.seed << '\n'
        << std::quoted(s.code_version) << '\n' << s.rng << '\n';
    for (const auto* array : {&s.x, &s.y, &s.z, &s.vx, &s.vy, &s.vz}) {
        for (double value : *array) out << value << ' ';
        out << '\n';
    }
    for (int species : s.species) out << species << ' ';
    out << '\n';
    out.close();
    require(static_cast<bool>(out), "Checkpoint write failed");
}

State load_checkpoint(const std::string& path) {
    std::ifstream in(path);
    require(static_cast<bool>(in), "Cannot open checkpoint for reading");
    in.imbue(std::locale::classic());
    State s;
    std::string header;
    std::size_t N = 0, proton_count = 0;
    double saved_L = 0.0;
    in >> header >> N >> proton_count;
    require(in && header == "M1_TEXT_V1", "Invalid checkpoint header or version");
    require(N > 0 && N <= 1000 && proton_count <= N, "Invalid checkpoint counts");
    in >> s.density >> saved_L >> s.proton_fraction
       >> s.cutoff >> s.dt >> s.time >> s.step >> s.seed
       >> std::quoted(s.code_version) >> s.rng;
    for (auto* array : {&s.x, &s.y, &s.z, &s.vx, &s.vy, &s.vz}) {
        array->resize(N);
        for (double& value : *array) in >> value;
    }
    s.species.resize(N);
    for (int& species : s.species) in >> species;
    require(static_cast<bool>(in), "Truncated or malformed checkpoint");
    std::string extra;
    require(!(in >> extra), "Unexpected trailing checkpoint data");
    validate(s);
    require(same_bits(saved_L, s.box_length()), "Saved L disagrees with N/density");
    require(static_cast<std::size_t>(std::count(s.species.begin(), s.species.end(), 1))
                == proton_count, "Proton count disagrees with species");
    return s;
}

void test_checkpoint(const State& a, const State& b) {
    require(a.size() == b.size() && a.step == b.step && a.seed == b.seed &&
                a.code_version == b.code_version && a.species == b.species &&
                a.rng == b.rng, "Checkpoint metadata or RNG changed");
    for (const auto& pair : {std::pair{a.density, b.density},
                            std::pair{a.proton_fraction, b.proton_fraction},
                            std::pair{a.cutoff, b.cutoff}, std::pair{a.dt, b.dt},
                            std::pair{a.time, b.time},
                            std::pair{a.box_length(), b.box_length()}})
        require(same_bits(pair.first, pair.second), "Checkpoint scalar bits changed");
    for (const auto& pair : {std::pair{&a.x, &b.x}, std::pair{&a.y, &b.y},
                            std::pair{&a.z, &b.z}, std::pair{&a.vx, &b.vx},
                            std::pair{&a.vy, &b.vy}, std::pair{&a.vz, &b.vz}})
        for (std::size_t i = 0; i < a.size(); ++i)
            require(same_bits((*pair.first)[i], (*pair.second)[i]),
                    "Checkpoint particle bits changed");
}

void test_geometry() {
    constexpr double L = 20.0;
    near(distance(19, 0, 0, 0, 0, 0, L), 1.0);                 // Face.
    near(distance(19, 19, 0, 0, 0, 0, L), std::sqrt(2.0));     // Edge.
    near(distance(19, 19, 19, 0, 0, 0, L), std::sqrt(3.0));    // Corner.
    near(distance(10, 10, 10, 0, 0, 0, L), 10 * std::sqrt(3.0));
    near(minimum_image(61, L), 1.0);
    near(minimum_image(-61, L), -1.0);
    near(std::abs(minimum_image(10, L)), 10.0);
    near(minimum_image(10, L), -minimum_image(-10, L));
    near(wrap(-41.25, L), 18.75);
    near(wrap(61.25, L), 1.25);
    near(wrap(20.0, L), 0.0);
    for (double original : {-41.25, -0.25, 0.0, 19.75, 20.0, 61.25}) {
        const double image = std::floor(original / L);
        require(same_bits(original, wrap(original, L) + L * image),
                "Constructed wrap/unwrap check failed");
    }
}

int main(int argc, char** argv) try {
    require(argc <= 2, "Usage: ./checkpoints/m1_box_checkpoint [checkpoint_path]");
    const std::string path = (argc == 2) ? argv[1] : "./checkpoints/m1_checkpoint.txt";
    test_geometry();
    std::cout << "PASS: face, edge, corner, half-box, multi-box and wrapping checks\n";

    const State state = make_test_state();
    const double L = state.box_length();
    const double bound = L * std::sqrt(3.0) / 2.0;
    double maximum = 0.0;
    for (std::size_t i = 0; i < state.size(); ++i)
        for (std::size_t j = i + 1; j < state.size(); ++j) {
            const double d = distance(state.x[i], state.y[i], state.z[i],
                                      state.x[j], state.y[j], state.z[j], L);
            require(d <= bound + 1e-12, "3D minimum-image distance exceeds bound");
            maximum = std::max(maximum, d);
        }
    std::cout << "PASS: all particle pairs satisfy the 3D distance bound\n"
              << "N=" << state.size() << " L=" << L << " max_distance=" << maximum
              << " bound=" << bound << '\n';

    State invalid = state;
    invalid.cutoff = L / 2.0 + 1.0;
    bool rejected = false;
    try { validate(invalid); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "Invalid cutoff was accepted");
    State boundary = state;
    boundary.cutoff = L / 2.0;
    validate(boundary);
    std::cout << "PASS: cutoff > L/2 rejected; cutoff = L/2 accepted\n";

    save_checkpoint(state, path);
    test_checkpoint(state, load_checkpoint(path));
    std::cout << "PASS: state, parameters and RNG round-trip bit-identically\n"
              << "Saved " << path << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}