// Units: positions in fm, density in fm^-3, velocities in c, time in fm/c.
// All doubles use max_digits10 and the classic locale for lossless round-trip.

// Build and run from the repository root: make tests

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <numbers>
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
    struct ModelParameters {
        double a = 110.0, b = -26.0, c = 24.0;
        double width = 1.25, charge = 1.44, lambda = 10.0, mass = 939.0;
    } model;
    double initial_temperature = 1.0;
    std::string thermostat = "nve";
    double dt = 0.5;
    double time = 0.0;
    std::uint64_t step = 0;
    std::uint64_t seed = 20261006;
    std::string code_version = "M3-v1";
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
    const auto& p = s.model;
    for (double value : {p.a, p.b, p.c, p.width, p.charge, p.lambda, p.mass})
        require(std::isfinite(value), "Nonfinite model parameter");
    require(p.width > 0.0 && p.charge >= 0.0 && p.lambda > 0.0 && p.mass > 0.0,
            "Invalid model parameter");
    require(std::isfinite(s.initial_temperature) && s.initial_temperature >= 0.0,
            "Invalid initial temperature");
    require(s.thermostat == "nve", "Only nve is implemented");
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
    require(static_cast<std::size_t>(std::count(s.species.begin(), s.species.end(), 1)) ==
                static_cast<std::size_t>(std::floor(s.proton_fraction * s.size())),
            "Proton count disagrees with proton fraction");
}

struct PairParameters {
    double a, b, k, cutoff_potential, cutoff_force;
};

struct Model {
    State::ModelParameters parameters;
    double cutoff;
    std::array<PairParameters, 4> pairs; // nn, np, pn, pp.
};

double raw_potential(double r, const PairParameters& pair, const Model& model) {
    const double r2 = r * r;
    const double width = model.parameters.width;
    const double nuclear = pair.a * std::exp(-r2 / width) +
                           pair.b * std::exp(-r2 / (2.0 * width));
    if (pair.k == 0.0) return nuclear;
    require(r > 0.0, "Coincident protons have divergent Coulomb energy");
    return nuclear + pair.k * std::exp(-r / model.parameters.lambda) / r;
}

double raw_radial_force(double r, const PairParameters& pair, const Model& model) {
    const double r2 = r * r;
    const double width = model.parameters.width;
    const double nuclear = (2.0 * pair.a * r / width) * std::exp(-r2 / width) +
                           (pair.b * r / width) * std::exp(-r2 / (2.0 * width));
    if (pair.k == 0.0) return nuclear;
    require(r > 0.0, "Coincident protons have divergent Coulomb force");
    return nuclear + pair.k * std::exp(-r / model.parameters.lambda) *
                     (1.0 / (model.parameters.lambda * r) + 1.0 / r2);
}

Model make_model(const State& s) {
    Model model{s.model, s.cutoff, {}};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            const double tau_product = (i == j) ? 1.0 : -1.0;
            PairParameters pair{s.model.a, s.model.b + s.model.c * tau_product,
                                (i == 1 && j == 1) ? s.model.charge : 0.0, 0.0, 0.0};
            pair.cutoff_potential = raw_potential(s.cutoff, pair, model);
            pair.cutoff_force = raw_radial_force(s.cutoff, pair, model);
            model.pairs[2 * i + j] = pair;
        }
    }
    return model;
}

double pair_potential(double r, const PairParameters& pair, const Model& model) {
    if (r >= model.cutoff) return 0.0;
    // V_sf(r) = V(r) - V(rc) + (r - rc) F(rc), with F = -dV/dr.
    return raw_potential(r, pair, model) - pair.cutoff_potential +
           (r - model.cutoff) * pair.cutoff_force;
}

double pair_radial_force(double r, const PairParameters& pair, const Model& model) {
    if (r >= model.cutoff) return 0.0;
    return raw_radial_force(r, pair, model) - pair.cutoff_force;
}

struct Forces {
    std::vector<double> x, y, z;
    double potential = 0.0;
};

Forces calculate_forces(const State& s) {
    const Model model = make_model(s);
    Forces result{std::vector<double>(s.size(), 0.0), std::vector<double>(s.size(), 0.0),
                  std::vector<double>(s.size(), 0.0), 0.0};
    const double L = s.box_length();
    const double cutoff2 = s.cutoff * s.cutoff;
    for (std::size_t i = 0; i < s.size(); ++i) {
        for (std::size_t j = i + 1; j < s.size(); ++j) {
            const double dx = minimum_image(s.x[i] - s.x[j], L);
            const double dy = minimum_image(s.y[i] - s.y[j], L);
            const double dz = minimum_image(s.z[i] - s.z[j], L);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= cutoff2) continue;
            const PairParameters& pair = model.pairs[2 * s.species[i] + s.species[j]];
            const double r = std::sqrt(r2);
            result.potential += pair_potential(r, pair, model);
            if (r2 == 0.0) continue; // Finite nuclear overlap has no direction.
            const double factor = pair_radial_force(r, pair, model) / r;
            const double fx = factor * dx, fy = factor * dy, fz = factor * dz;
            result.x[i] += fx; result.x[j] -= fx;
            result.y[i] += fy; result.y[j] -= fy;
            result.z[i] += fz; result.z[j] -= fz;
        }
    }
    return result;
}

double kinetic_energy(const State& s) {
    double energy = 0.0;
    for (std::size_t i = 0; i < s.size(); ++i)
        energy += 0.5 * s.model.mass *
                  (s.vx[i] * s.vx[i] + s.vy[i] * s.vy[i] + s.vz[i] * s.vz[i]);
    return energy;
}

double total_energy(const State& s) {
    validate(s);
    return kinetic_energy(s) + calculate_forces(s).potential;
}

double thomas_fermi_lambda(double density, double proton_fraction) {
    require(std::isfinite(density) && density > 0.0 &&
                std::isfinite(proton_fraction) && proton_fraction > 0.0 &&
                proton_fraction <= 1.0, "Thomas-Fermi screening needs positive n and Yp");
    constexpr double hbar_c = 197.327;
    constexpr double fine_structure = 1.0 / 137.036;
    constexpr double electron_mass = 0.511;
    const double kf = hbar_c * std::cbrt(3.0 * std::numbers::pi *
                                        std::numbers::pi * density * proton_fraction);
    return hbar_c * std::sqrt(std::numbers::pi) /
           (std::sqrt(4.0 * fine_structure * kf) *
            std::pow(kf * kf + electron_mass * electron_mass, 0.25));
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

State make_initial_state(std::size_t count, double density, double proton_fraction,
                         double temperature, double dt, double cutoff, double lambda,
                         std::uint64_t seed) {
    require(count >= 2 && count <= 1000, "CPU development run needs 2 <= N <= 1000");
    require(std::isfinite(temperature) && temperature > 0.0,
            "Initial temperature must be positive");
    require(std::isfinite(proton_fraction) && proton_fraction >= 0.0 &&
                proton_fraction <= 1.0, "Invalid initial proton fraction");
    State s;
    s.density = density;
    s.proton_fraction = proton_fraction;
    s.initial_temperature = temperature;
    s.dt = dt;
    s.cutoff = cutoff;
    s.model.lambda = lambda;
    s.seed = seed;
    for (auto* array : {&s.x, &s.y, &s.z, &s.vx, &s.vy, &s.vz})
        array->resize(count);
    s.species.assign(count, 0);
    s.rng.seed(seed);
    const double L = s.box_length();
    require(std::isfinite(L) && L > 0.0, "Invalid initial box length");
    auto uniform = [&]() {
        return static_cast<double>(s.rng() >> 11) * 0x1.0p-53 * L;
    };
    const auto protons = static_cast<std::size_t>(std::floor(proton_fraction * count));
    require(protons <= count, "Invalid initial proton count");
    std::fill_n(s.species.begin(), protons, 1);
    std::shuffle(s.species.begin(), s.species.end(), s.rng);
    std::normal_distribution<double> gaussian(0.0,
                                               std::sqrt(temperature / s.model.mass));
    for (std::size_t i = 0; i < count; ++i) {
        s.x[i] = uniform(); s.y[i] = uniform(); s.z[i] = uniform();
        s.vx[i] = gaussian(s.rng);
        s.vy[i] = gaussian(s.rng);
        s.vz[i] = gaussian(s.rng);
    }
    for (auto* component : {&s.vx, &s.vy, &s.vz}) {
        double mean = 0.0;
        for (double value : *component) mean += value;
        mean /= static_cast<double>(count);
        for (double& value : *component) value -= mean;
    }
    const double target_kinetic = 0.5 * (3.0 * count - 3.0) * temperature;
    const double scale = std::sqrt(target_kinetic / kinetic_energy(s));
    for (auto* component : {&s.vx, &s.vy, &s.vz})
        for (double& value : *component) value *= scale;
    validate(s);
    return s;
}

std::array<double, 3> momentum(const State& s) {
    std::array<double, 3> result{};
    for (std::size_t i = 0; i < s.size(); ++i) {
        result[0] += s.model.mass * s.vx[i];
        result[1] += s.model.mass * s.vy[i];
        result[2] += s.model.mass * s.vz[i];
    }
    return result;
}

void nve_step(State& s, Forces& forces) {
    require(s.thermostat == "nve", "Velocity Verlet requires nve mode");
    const double half_kick = s.dt / (2.0 * s.model.mass);
    const double L = s.box_length();
    for (std::size_t i = 0; i < s.size(); ++i) {
        s.vx[i] += half_kick * forces.x[i];
        s.vy[i] += half_kick * forces.y[i];
        s.vz[i] += half_kick * forces.z[i];
        s.x[i] = wrap(s.x[i] + s.dt * s.vx[i], L);
        s.y[i] = wrap(s.y[i] + s.dt * s.vy[i], L);
        s.z[i] = wrap(s.z[i] + s.dt * s.vz[i], L);
    }
    Forces next = calculate_forces(s); // One new force evaluation per step.
    for (std::size_t i = 0; i < s.size(); ++i) {
        s.vx[i] += half_kick * next.x[i];
        s.vy[i] += half_kick * next.y[i];
        s.vz[i] += half_kick * next.z[i];
    }
    forces = std::move(next);
    ++s.step;
    s.time += s.dt;
    validate(s);
}

void test_model() {
    State s = make_test_state();
    const Model model = make_model(s);
    require(model.pairs[0].b == -2.0 && model.pairs[1].b == -50.0 &&
                model.pairs[2].b == -50.0 && model.pairs[3].b == -2.0 &&
                model.pairs[0].k == 0.0 && model.pairs[1].k == 0.0 &&
                model.pairs[3].k == 1.44, "Incorrect pair-parameter table");
    constexpr double h = 1e-5;
    std::mt19937_64 rng(20261117);
    std::uniform_real_distribution<double> separation(0.1 + h, s.cutoff - 0.1);
    double worst_relative = 0.0;
    for (int kind : {0, 1, 3}) {
        const PairParameters& pair = model.pairs[kind];
        for (int sample = 0; sample < 20; ++sample) {
            const double r = separation(rng);
            const double analytic = pair_radial_force(r, pair, model);
            const double finite_difference =
                -(pair_potential(r + h, pair, model) -
                  pair_potential(r - h, pair, model)) / (2.0 * h);
            const double relative = std::abs(analytic - finite_difference) /
                                    std::abs(analytic);
            require(std::isfinite(relative) && relative < 1e-8,
                    "Force does not match the potential gradient");
            worst_relative = std::max(worst_relative, relative);
        }
        require(pair_potential(model.cutoff, pair, model) == 0.0 &&
                    pair_radial_force(model.cutoff, pair, model) == 0.0,
                "Potential or force is nonzero at cutoff");
        require(std::abs(pair_potential(model.cutoff - 1e-6, pair, model)) < 1e-8 &&
                    std::abs(pair_radial_force(model.cutoff - 1e-6, pair, model)) < 1e-6,
                "Potential or force is discontinuous at cutoff");
    }
    State two = s;
    for (auto* array : {&two.x, &two.y, &two.z, &two.vx, &two.vy, &two.vz})
        array->resize(2, 0.0);
    two.species = {0, 1};
    two.proton_fraction = 0.5;
    two.density = 2.0 / (20.0 * 20.0 * 20.0);
    two.x = {9.0, 11.0};
    two.y = {10.0, 10.5};
    two.z = {10.0, 10.0};
    validate(two);
    const Forces f = calculate_forces(two);
    for (const auto* component : {&f.x, &f.y, &f.z})
        require((*component)[0] == -(*component)[1],
                "Pair force violates Newton's third law");
    const double dx = minimum_image(two.x[0] - two.x[1], two.box_length());
    const double dy = minimum_image(two.y[0] - two.y[1], two.box_length());
    require(std::abs(f.x[0] * dy - f.y[0] * dx) < 1e-12 && f.z[0] == 0.0,
            "Pair force is not parallel to separation");
    const std::array<std::array<double, 3>, 3> expected{{
        {26.6, 23.3, 21.1}, {15.6, 13.6, 12.3}, {12.4, 10.8, 9.8}}};
    const std::array<double, 3> densities{0.01, 0.05, 0.10};
    for (std::size_t i = 0; i < expected.size(); ++i)
        for (std::size_t j = 0; j < expected[i].size(); ++j)
            require(std::abs(thomas_fermi_lambda(densities[i],
                                                  0.2 + 0.1 * static_cast<double>(j)) -
                             expected[i][j]) < 0.15,
                    "Thomas-Fermi screening table disagrees with reference");
    std::cout << "PASS: pair table, 60 force gradients (worst relative "
              << worst_relative << "), cutoff continuity, Newton's law, screening\n";
}

void save_checkpoint(const State& s, const std::string& path) {
    validate(s);
    std::ofstream out(path);
    require(static_cast<bool>(out), "Cannot open checkpoint for writing");
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "PASTA_TEXT_V2\n" << s.size() << ' '
        << std::count(s.species.begin(), s.species.end(), 1) << '\n'
        << s.density << ' ' << s.box_length() << ' ' << s.proton_fraction << '\n'
        << s.cutoff << ' ' << s.dt << ' ' << s.time << ' '
        << s.step << ' ' << s.seed << '\n'
        << s.model.a << ' ' << s.model.b << ' ' << s.model.c << ' '
        << s.model.width << ' ' << s.model.charge << ' ' << s.model.lambda << ' '
        << s.model.mass << ' ' << s.initial_temperature << '\n'
        << std::quoted(s.thermostat) << ' ' << std::quoted(s.code_version)
        << '\n' << s.rng << '\n';
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
    require(in && (header == "M1_TEXT_V1" || header == "PASTA_TEXT_V2"),
            "Invalid checkpoint header or version");
    require(N > 0 && N <= 1000 && proton_count <= N, "Invalid checkpoint counts");
    in >> s.density >> saved_L >> s.proton_fraction
       >> s.cutoff >> s.dt >> s.time >> s.step >> s.seed;
    if (header == "PASTA_TEXT_V2") {
        in >> s.model.a >> s.model.b >> s.model.c >> s.model.width
           >> s.model.charge >> s.model.lambda >> s.model.mass
           >> s.initial_temperature >> std::quoted(s.thermostat);
    }
    in >> std::quoted(s.code_version) >> s.rng;
    require(static_cast<bool>(in), "Truncated checkpoint metadata");
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
                a.code_version == b.code_version && a.thermostat == b.thermostat &&
                a.species == b.species &&
                a.rng == b.rng, "Checkpoint metadata or RNG changed");
    for (const auto& pair : {std::pair{a.density, b.density},
                            std::pair{a.proton_fraction, b.proton_fraction},
                            std::pair{a.cutoff, b.cutoff}, std::pair{a.dt, b.dt},
                            std::pair{a.time, b.time},
                            std::pair{a.initial_temperature, b.initial_temperature},
                            std::pair{a.model.a, b.model.a},
                            std::pair{a.model.b, b.model.b},
                            std::pair{a.model.c, b.model.c},
                            std::pair{a.model.width, b.model.width},
                            std::pair{a.model.charge, b.model.charge},
                            std::pair{a.model.lambda, b.model.lambda},
                            std::pair{a.model.mass, b.model.mass},
                            std::pair{a.box_length(), b.box_length()}})
        require(same_bits(pair.first, pair.second), "Checkpoint scalar bits changed");
    for (const auto& pair : {std::pair{&a.x, &b.x}, std::pair{&a.y, &b.y},
                            std::pair{&a.z, &b.z}, std::pair{&a.vx, &b.vx},
                            std::pair{&a.vy, &b.vy}, std::pair{&a.vz, &b.vz}})
        for (std::size_t i = 0; i < a.size(); ++i)
            require(same_bits((*pair.first)[i], (*pair.second)[i]),
                    "Checkpoint particle bits changed");
    const double original_energy = total_energy(a);
    const double restored_energy = total_energy(b);
    require(std::isfinite(original_energy) && std::isfinite(restored_energy),
            "Nonfinite checkpoint energy");
    require(same_bits(original_energy, restored_energy),
            "Checkpoint total-energy bits changed");
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
    std::mt19937_64 rng(917);
    for (double box : {4.0, 20.0, 37.5}) {
        std::uniform_real_distribution<double> coordinate(-4.0 * box, 4.0 * box);
        const double bound = box * std::sqrt(3.0) / 2.0;
        for (int sample = 0; sample < 1000; ++sample) {
            const double x1 = coordinate(rng), y1 = coordinate(rng), z1 = coordinate(rng);
            const double x2 = coordinate(rng), y2 = coordinate(rng), z2 = coordinate(rng);
            require(distance(x1, y1, z1, x2, y2, z2, box) <= bound + 1e-12,
                    "Random minimum-image distance exceeds bound");
            for (double position : {x1, y1, z1, x2, y2, z2}) {
                const double wrapped = wrap(position, box);
                require(wrapped >= 0.0 && wrapped < box,
                        "Wrapped position outside [0,L)");
            }
        }
    }
}

bool is_invalid(const State& s) {
    try { validate(s); }
    catch (const std::runtime_error&) { return true; }
    return false;
}

void test_validation(const State& state) {
    const double L = state.box_length();
    State invalid = state;
    invalid.cutoff = 0.0;
    require(is_invalid(invalid), "Zero cutoff was accepted");
    invalid.cutoff = L / 2.0 + 1.0;
    require(is_invalid(invalid), "Cutoff above L/2 was accepted");
    invalid = state;
    invalid.x[0] = -0.1;
    require(is_invalid(invalid), "Negative position was accepted");
    invalid.x[0] = L;
    require(is_invalid(invalid), "Position at L was accepted");
    invalid = state;
    invalid.species[0] = 1 - invalid.species[0];
    require(is_invalid(invalid), "Wrong proton count was accepted");
    State boundary = state;
    boundary.cutoff = L / 2.0;
    validate(boundary);
}

void test_m3_fast(const std::string& checkpoint_path) {
    State initial = make_initial_state(64, 0.025, 0.3, 1.0, 0.25, 6.0, 10.0, 314159);
    const double measured_temperature = 2.0 * kinetic_energy(initial) /
                                        (3.0 * initial.size() - 3.0);
    require(std::abs(measured_temperature - 1.0) < 1e-12,
            "Initial velocity temperature is wrong");
    const auto initial_momentum = momentum(initial);
    double momentum_scale = 0.0;
    for (std::size_t i = 0; i < initial.size(); ++i)
        momentum_scale += initial.model.mass *
                          std::sqrt(initial.vx[i] * initial.vx[i] +
                                    initial.vy[i] * initial.vy[i] +
                                    initial.vz[i] * initial.vz[i]);
    State continuous = initial;
    Forces continuous_forces = calculate_forces(continuous);
    double maximum_momentum_error = 0.0;
    for (int step = 0; step < 40; ++step) {
        nve_step(continuous, continuous_forces);
        const auto current = momentum(continuous);
        for (std::size_t axis = 0; axis < 3; ++axis)
            maximum_momentum_error = std::max(maximum_momentum_error,
                std::abs(current[axis] - initial_momentum[axis]) / momentum_scale);
        require(std::isfinite(kinetic_energy(continuous) + continuous_forces.potential),
                "Nonfinite NVE energy");
    }
    require(maximum_momentum_error < 1e-14, "NVE momentum drift is too large");
    State interrupted = initial;
    Forces interrupted_forces = calculate_forces(interrupted);
    for (int step = 0; step < 20; ++step) nve_step(interrupted, interrupted_forces);
    save_checkpoint(interrupted, checkpoint_path);
    State resumed = load_checkpoint(checkpoint_path);
    test_checkpoint(interrupted, resumed);
    Forces resumed_forces = calculate_forces(resumed);
    for (int step = 0; step < 20; ++step) nve_step(resumed, resumed_forces);
    test_checkpoint(continuous, resumed);
    std::cout << "PASS: NVE initialization, momentum (max relative drift "
              << maximum_momentum_error << "), and bit-identical restart\n";
}

void test_m3_slow(const std::string& csv_path) {
    constexpr std::size_t count = 200;
    constexpr double density = 0.05, proton_fraction = 0.3;
    constexpr double temperature = 1.0, cutoff = 7.0, lambda = 10.0;
    constexpr double duration = 64.0;
    constexpr std::uint64_t seed = 271828;
    const State initial = make_initial_state(count, density, proton_fraction,
                                              temperature, 0.5, cutoff, lambda, seed);
    const std::array<double, 8> timesteps{2.0, 1.0, 0.5, 0.25, 0.125,
                                           0.0625, 0.03125, 0.015625};
    std::array<double, 8> errors{};
    std::ofstream csv(csv_path);
    require(static_cast<bool>(csv), "Cannot open timestep sweep CSV");
    csv.imbue(std::locale::classic());
    csv << std::setprecision(std::numeric_limits<double>::max_digits10);
    csv << "N,density,proton_fraction,lambda,cutoff,temperature,seed,duration,dt,epsilon\n";
    const auto starting_momentum = momentum(initial);
    double momentum_scale = 0.0, momentum_error = 0.0;
    for (std::size_t i = 0; i < count; ++i)
        momentum_scale += initial.model.mass *
                          std::sqrt(initial.vx[i] * initial.vx[i] +
                                    initial.vy[i] * initial.vy[i] +
                                    initial.vz[i] * initial.vz[i]);
    for (std::size_t sample = 0; sample < timesteps.size(); ++sample) {
        State s = initial;
        s.dt = timesteps[sample];
        Forces forces = calculate_forces(s);
        const int steps = static_cast<int>(duration / s.dt);
        double mean = 0.0, sum_squared = 0.0;
        for (int step = 1; step <= steps; ++step) {
            nve_step(s, forces);
            const double energy = kinetic_energy(s) + forces.potential;
            require(std::isfinite(energy), "Nonfinite energy in timestep sweep");
            const double delta = energy - mean;
            mean += delta / step;
            sum_squared += delta * (energy - mean);
            if (timesteps[sample] == 0.5) {
                const auto current = momentum(s);
                for (std::size_t axis = 0; axis < 3; ++axis)
                    momentum_error = std::max(momentum_error,
                        std::abs(current[axis] - starting_momentum[axis]) /
                        momentum_scale);
            }
        }
        errors[sample] = std::sqrt(sum_squared / steps) / std::abs(mean);
        require(std::isfinite(errors[sample]) && errors[sample] > 0.0,
                "Invalid energy error in timestep sweep");
        csv << count << ',' << density << ',' << proton_fraction << ',' << lambda
            << ',' << cutoff << ',' << temperature << ',' << seed << ','
            << duration << ',' << s.dt << ',' << errors[sample] << '\n';
        std::cout << "M3 sweep: dt=" << s.dt << " epsilon=" << errors[sample] << '\n';
    }
    csv.close();
    require(static_cast<bool>(csv), "Cannot write timestep sweep CSV");
    double mean_x = 0.0, mean_y = 0.0;
    for (std::size_t i = 1; i < timesteps.size(); ++i) {
        mean_x += std::log(timesteps[i]);
        mean_y += std::log(errors[i]);
    }
    mean_x /= timesteps.size() - 1;
    mean_y /= timesteps.size() - 1;
    double covariance = 0.0, variance = 0.0;
    for (std::size_t i = 1; i < timesteps.size(); ++i) {
        const double x = std::log(timesteps[i]) - mean_x;
        covariance += x * (std::log(errors[i]) - mean_y);
        variance += x * x;
    }
    const double slope = covariance / variance;
    std::cout << "M3 sweep: log-log slope (dt <= 1)=" << slope
              << " momentum drift=" << momentum_error << '\n';
    require(std::abs(slope - 2.0) <= 0.05, "Energy-error slope is not 2.00 +/- 0.05");
    require(momentum_error < 1e-14, "N=200 momentum drift is too large");

    State reversible = make_initial_state(2, 2.0 / 8000.0, 0.5, 0.01,
                                           0.001, 8.0, 10.0, seed);
    reversible.x = {9.0, 11.0}; reversible.y = {10.0, 10.0};
    reversible.z = {10.0, 10.0};
    reversible.vx = {-0.01, 0.01}; reversible.vy = {0.0, 0.0};
    reversible.vz = {0.0, 0.0};
    validate(reversible);
    const State starting_state = reversible;
    Forces reversible_forces = calculate_forces(reversible);
    for (int step = 0; step < 10000; ++step)
        nve_step(reversible, reversible_forces);
    for (auto* component : {&reversible.vx, &reversible.vy, &reversible.vz})
        for (double& value : *component) value = -value;
    for (int step = 0; step < 10000; ++step)
        nve_step(reversible, reversible_forces);
    double maximum_position_error = 0.0;
    for (std::size_t i = 0; i < reversible.size(); ++i)
        for (const auto& pair : {std::pair{&reversible.x, &starting_state.x},
                                 std::pair{&reversible.y, &starting_state.y},
                                 std::pair{&reversible.z, &starting_state.z}})
            maximum_position_error = std::max(maximum_position_error,
                std::abs(minimum_image((*pair.first)[i] - (*pair.second)[i],
                                       reversible.box_length())));
    std::cout << "M3 reversal: 10000 + 10000 steps, max position error="
              << maximum_position_error << " fm\n";
    require(maximum_position_error < 1e-9, "NVE reversal error exceeds 1e-9 fm");
    State periodic = make_initial_state(2, 2.0 / 8000.0, 0.0, 0.01,
                                         0.1, 8.0, 10.0, seed);
    periodic.x = {0.1, 10.1}; periodic.y = {10.0, 10.0};
    periodic.z = {10.0, 10.0};
    periodic.vx = {0.02, 0.02}; periodic.vy = {0.0, 0.0};
    periodic.vz = {0.0, 0.0};
    validate(periodic);
    const State periodic_start = periodic;
    Forces periodic_forces = calculate_forces(periodic);
    for (int step = 0; step < 10000; ++step) nve_step(periodic, periodic_forces);
    for (double& velocity : periodic.vx) velocity = -velocity;
    for (int step = 0; step < 10000; ++step) nve_step(periodic, periodic_forces);
    double periodic_error = 0.0;
    for (std::size_t i = 0; i < periodic.size(); ++i)
        periodic_error = std::max(periodic_error,
            std::abs(minimum_image(periodic.x[i] - periodic_start.x[i],
                                   periodic.box_length())));
    std::cout << "M3 periodic reversal: max position error=" << periodic_error
              << " fm\n";
    require(periodic_error < 1e-9, "Periodic reversal error exceeds 1e-9 fm");
    std::cout << "PASS: M3 timestep slope, momentum, and reversibility\n";
}

std::map<std::string, std::string> parse_options(int argc, char** argv, int start) {
    require((argc - start) % 2 == 0, "Every option needs a value");
    std::map<std::string, std::string> options;
    for (int i = start; i < argc; i += 2) {
        const std::string key = argv[i];
        require(key.rfind("--", 0) == 0, "Expected --option value pairs");
        require(options.emplace(key, argv[i + 1]).second, "Duplicate option");
    }
    return options;
}

void check_options(const std::map<std::string, std::string>& options,
                   const std::vector<std::string>& allowed) {
    for (const auto& entry : options)
        require(std::find(allowed.begin(), allowed.end(), entry.first) != allowed.end(),
                "Unknown option");
}

std::string option_text(const std::map<std::string, std::string>& options,
                        const std::string& key, const std::string& fallback) {
    const auto found = options.find(key);
    return found == options.end() ? fallback : found->second;
}

double parse_number(const std::string& value) {
    std::size_t used = 0;
    const double result = std::stod(value, &used);
    require(used == value.size() && std::isfinite(result), "Invalid real-number option");
    return result;
}

std::uint64_t parse_integer(const std::string& value) {
    require(!value.empty() && value[0] != '-', "Invalid nonnegative integer option");
    std::size_t used = 0;
    const auto result = std::stoull(value, &used);
    require(used == value.size(), "Invalid integer option");
    return result;
}

void write_energy_row(std::ofstream& csv, const State& s, const Forces& forces) {
    const auto p = momentum(s);
    const double kinetic = kinetic_energy(s);
    csv << s.step << ',' << s.time << ',' << s.size() << ',' << s.density << ','
        << s.proton_fraction << ','
        << std::count(s.species.begin(), s.species.end(), 1) << ','
        << s.dt << ',' << s.initial_temperature << ',' << s.seed << ','
        << s.cutoff << ',' << s.model.a << ',' << s.model.b << ',' << s.model.c
        << ',' << s.model.width << ',' << s.model.charge << ',' << s.model.lambda
        << ',' << s.model.mass << ',' << s.thermostat << ',' << s.code_version
        << ',' << kinetic << ','
        << forces.potential << ',' << kinetic + forces.potential << ','
        << p[0] << ',' << p[1] << ',' << p[2] << '\n';
}

void run_simulation(State& s, std::uint64_t steps, std::uint64_t log_every,
                    const std::string& checkpoint_path, const std::string& csv_path) {
    require(steps > 0 && log_every > 0, "Steps and log interval must be positive");
    require(steps <= std::numeric_limits<std::uint64_t>::max() - s.step,
            "Step counter would overflow");
    validate(s);
    Forces forces = calculate_forces(s);
    std::ofstream csv(csv_path);
    require(static_cast<bool>(csv), "Cannot open energy CSV");
    csv.imbue(std::locale::classic());
    csv << std::setprecision(std::numeric_limits<double>::max_digits10);
    csv << "step,time,N,density,proton_fraction,protons,dt,initial_temperature,"
           "seed,cutoff,a,b,c,width,charge,lambda,mass,thermostat,code_version,kinetic,"
           "potential,total,px,py,pz\n";
    write_energy_row(csv, s, forces);
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 1; i <= steps; ++i) {
        nve_step(s, forces);
        if (i % log_every == 0 || i == steps) write_energy_row(csv, s, forces);
    }
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    csv.close();
    require(static_cast<bool>(csv), "Energy CSV write failed");
    save_checkpoint(s, checkpoint_path);
    std::cout << std::setprecision(17)
              << "N=" << s.size() << " step=" << s.step << " time=" << s.time
              << " dt=" << s.dt << " lambda=" << s.model.lambda << '\n'
              << "Final energy=" << kinetic_energy(s) + forces.potential << " MeV\n"
              << "Wall time=" << wall_seconds << " s; time/step="
              << wall_seconds / steps << " s; particle-steps/s="
              << static_cast<double>(s.size()) * steps / wall_seconds << '\n'
              << "Checkpoint: " << checkpoint_path << '\n'
              << "Energy CSV: " << csv_path << '\n';
}

void run_cli(int argc, char** argv) {
    const std::string command = argv[1];
    const auto options = parse_options(argc, argv, 2);
    if (command == "simulate") {
        check_options(options, {"--n", "--density", "--yp", "--temperature",
                                "--dt", "--cutoff", "--lambda", "--seed",
                                "--steps", "--checkpoint", "--csv", "--log-every"});
        const auto count = parse_integer(option_text(options, "--n", "200"));
        require(count >= 2 && count <= 1000, "CPU development run needs 2 <= N <= 1000");
        const double density = parse_number(option_text(options, "--density", "0.05"));
        const double yp = parse_number(option_text(options, "--yp", "0.3"));
        const double temperature = parse_number(option_text(options, "--temperature", "1"));
        const double dt = parse_number(option_text(options, "--dt", "0.5"));
        const double cutoff = parse_number(option_text(options, "--cutoff", "7"));
        const std::string screening = option_text(options, "--lambda", "10");
        const double lambda = screening == "tf" ? thomas_fermi_lambda(density, yp) :
                                                 parse_number(screening);
        const auto seed = parse_integer(option_text(options, "--seed", "20261117"));
        const auto steps = parse_integer(option_text(options, "--steps", "100"));
        const auto log_every = parse_integer(option_text(options, "--log-every", "1"));
        State s = make_initial_state(static_cast<std::size_t>(count), density, yp,
                                     temperature, dt, cutoff, lambda, seed);
        run_simulation(s, steps, log_every,
                       option_text(options, "--checkpoint", "checkpoints/run.txt"),
                       option_text(options, "--csv", "checkpoints/energy.csv"));
    } else if (command == "resume") {
        check_options(options, {"--input", "--steps", "--checkpoint", "--csv",
                                "--log-every"});
        const std::string input = option_text(options, "--input", "");
        require(!input.empty(), "resume requires --input checkpoint_path");
        State s = load_checkpoint(input);
        const auto steps = parse_integer(option_text(options, "--steps", "100"));
        const auto log_every = parse_integer(option_text(options, "--log-every", "1"));
        run_simulation(s, steps, log_every,
                       option_text(options, "--checkpoint", "checkpoints/resumed.txt"),
                       option_text(options, "--csv", "checkpoints/resumed_energy.csv"));
    } else {
        throw std::runtime_error("Unknown command; use simulate or resume");
    }
}

int main(int argc, char** argv) try {
    if (argc >= 2 && (std::string(argv[1]) == "simulate" ||
                      std::string(argv[1]) == "resume")) {
        run_cli(argc, argv);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "test-slow") {
        require(argc <= 3, "Usage: ./bin/exec test-slow [csv_path]");
        test_m3_slow(argc == 3 ? argv[2] : "checkpoints/m3_energy_scaling.csv");
        return 0;
    }
    require(argc <= 2, "Usage: ./bin/exec [checkpoint_path] or simulate/resume/test-slow");
    const std::string path = (argc == 2) ? argv[1] : "./checkpoints/m1_checkpoint.txt";
    test_geometry();
    std::cout << "PASS: face, edge, corner, half-box, multi-box and random geometry checks\n";

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

    test_validation(state);
    std::cout << "PASS: cutoff, position and proton-count invariants\n";

    test_model();

    save_checkpoint(state, path);
    test_checkpoint(state, load_checkpoint(path));
    std::cout << "PASS: state, parameters, RNG and total energy round-trip bit-identically\n"
              << "Saved " << path << '\n';
    test_m3_fast(path + ".restart");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
