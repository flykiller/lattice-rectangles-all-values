#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

// Exact O(N^(3/2)) all-values baseline with O(N log N) range-add recovery.
// Benchmark build: g++ -std=c++20 -O3 -DNDEBUG -march=native
//                  all_values_previous.cpp -o all_values_previous.exe
// Run: all_values_previous.exe FIRST_POWER LAST_POWER REPEATS
// Test: all_values_previous.exe --self-test

using i128 = __int128_t;
using u128 = __uint128_t;
using std::size_t;
using std::string;
using std::vector;

constexpr int MAX_N = 1 << 30;

static inline string to_string_i128(i128 x) {
    if (x == 0) return "0";
    bool neg = (x < 0);
    u128 y = neg ? (u128)(-(x + 1)) + 1 : (u128)x;
    string s;
    while (y != 0) {
        s.push_back(char('0' + (unsigned)(y % 10)));
        y /= 10;
    }
    if (neg) s.push_back('-');
    std::reverse(s.begin(), s.end());
    return s;
}

static inline long long isqrt_ll(long long n) {
    long long r = (long long)std::sqrt((long double)n);
    while ((i128)(r + 1) * (r + 1) <= n) ++r;
    while ((i128)r * r > n) --r;
    return r;
}

static vector<int8_t> mobius_sieve(int n) {
    vector<int8_t> mu((size_t)n + 1, 0);
    vector<uint8_t> comp((size_t)n + 1, 0);
    vector<int> primes;
    if (n >= 16) primes.reserve((size_t)(n / std::log((double)n) * 1.3));
    if (n >= 1) mu[1] = 1;

    for (int i = 2; i <= n; ++i) {
        if (!comp[(size_t)i]) {
            primes.push_back(i);
            mu[(size_t)i] = -1;
        }
        for (int p : primes) {
            long long v = 1LL * i * p;
            if (v > n) break;
            comp[(size_t)v] = 1;
            if (i % p == 0) {
                mu[(size_t)v] = 0;
                break;
            }
            mu[(size_t)v] = (int8_t)(-mu[(size_t)i]);
        }
    }
    return mu;
}

static inline i128 F0_closed(i128 n) {
    return n * n * (n - 1) * (n - 1) / 4
         + n * (n - 1) * (n - 1) * (n - 2) / 12;
}

template<int DEG>
struct StepDiff {
    vector<i128> d0;
    vector<i128> d1;
    vector<i128> d2;

    explicit StepDiff(long long N) : d0((size_t)N + 1, 0) {
        if constexpr (DEG >= 1) d1.assign((size_t)N + 1, 0);
        if constexpr (DEG >= 2) d2.assign((size_t)N + 1, 0);
    }

    inline void add(long long N, long long step, long long Lbase,
                    long long lo, long long hi, i128 c0, i128 c1, i128 c2) {
        if (lo > hi) return;

        const long long r = Lbase % step;
        const long long shift = (Lbase - r) / step;
        const long long qlo = shift + lo;
        const long long qhi = shift + hi;
        const long long Lstart = r + step * qlo;
        if (Lstart > N) return;

        const i128 sh = (i128)shift;
        const i128 A2 = c2;
        const i128 A1 = c1 - 2 * c2 * sh;
        const i128 A0 = c0 - c1 * sh + c2 * sh * sh;

        const size_t st = (size_t)Lstart;
        d0[st] += A0;
        if constexpr (DEG >= 1) d1[st] += A1;
        if constexpr (DEG >= 2) d2[st] += A2;

        const long long Lafter = r + step * (qhi + 1);
        if (Lafter <= N) {
            const size_t af = (size_t)Lafter;
            d0[af] -= A0;
            if constexpr (DEG >= 1) d1[af] -= A1;
            if constexpr (DEG >= 2) d2[af] -= A2;
        }
    }

    inline void flush(long long N, long long step, vector<i128>& G) {
        for (long long r = 0; r < step; ++r) {
            i128 A0 = 0, A1 = 0, A2 = 0;
            long long q = 0;
            for (long long L = r; L <= N; L += step, ++q) {
                const size_t idx = (size_t)L;
                A0 += d0[idx];
                d0[idx] = 0;
                if constexpr (DEG >= 1) {
                    A1 += d1[idx];
                    d1[idx] = 0;
                }
                if constexpr (DEG >= 2) {
                    A2 += d2[idx];
                    d2[idx] = 0;
                }

                if (L == 0) continue;
                if constexpr (DEG == 0) {
                    if (A0 != 0) G[idx] += A0;
                } else if constexpr (DEG == 1) {
                    if (A0 != 0 || A1 != 0) {
                        G[idx] += A0 + A1 * (i128)q;
                    }
                } else {
                    if (A0 != 0 || A1 != 0 || A2 != 0) {
                        const i128 qq = (i128)q;
                        G[idx] += A0 + A1 * qq + A2 * qq * qq;
                    }
                }
            }
        }
    }
};

template<int C>
static inline void add_direct_value(vector<i128>& G, long long N, long long L, i128 value) {
    if (1 <= L && L <= N && value != 0) G[(size_t)L] += value;
}

template<int C>
static inline void add_diag_side(vector<i128>& G, long long N, long long L, long long cnt) {
    if constexpr (C == 0) {
        add_direct_value<C>(G, N, L, (i128)2 * cnt);
    } else if constexpr (C == 1) {
        add_direct_value<C>(G, N, L, (i128)4 * L * cnt);
    } else {
        add_direct_value<C>(G, N, L, (i128)2 * L * L * cnt);
    }
}



template<int C>
static vector<i128> build_G_component(long long N) {
    const long long B = isqrt_ll(N);
    vector<i128> G((size_t)N + 1, 0);
    StepDiff<C> diff(N);


    for (long long t = 1; t <= B; ++t) {
        const long long sMax = N / t;
        for (long long s = 3; s <= sMax; ++s) {
            add_diag_side<C>(G, N, t * s, (s - 1) >> 1);
        }
    }

    for (long long step = 1; step <= 2 * B; ++step) {
        bool has_updates = false;



        if (2 <= step && step <= B) {
            const long long a = step;
            for (long long b = 1; b < a; ++b) {
                const long long sum = a + b;
                const long long yMax = N / sum;
                for (long long y = 1; y <= yMax; ++y) {
                    const long long Cbase = sum * y;
                    const long long hi = (N - Cbase) / a;


                    if (y > B) {
                        if constexpr (C == 0) {
                            add_direct_value<C>(G, N, Cbase, 2);
                        } else if constexpr (C == 1) {
                            add_direct_value<C>(G, N, Cbase, (i128)4 * Cbase);
                        } else {
                            add_direct_value<C>(G, N, Cbase, (i128)2 * Cbase * Cbase);
                        }
                    }


                    const long long lo = std::max(1LL, B - y + 1);
                    if (lo <= hi) {
                        has_updates = true;
                        if constexpr (C == 0) {
                            diff.add(N, a, Cbase, lo, hi, 4, 0, 0);
                        } else if constexpr (C == 1) {

                            diff.add(N, a, Cbase, lo, hi, (i128)8 * Cbase, (i128)4 * sum, 0);
                        } else {

                            diff.add(N, a, Cbase, lo, hi,
                                     (i128)4 * Cbase * Cbase,
                                     (i128)4 * Cbase * sum,
                                     (i128)4 * a * b);
                        }
                    }
                }
            }
        }



        const long long xLo = step / 2 + 1;
        const long long xHi = std::min(B, step - 1);
        for (long long x = xLo; x <= xHi; ++x) {
            const long long y = step - x;
            if (y <= 0 || y >= x) continue;
            const long long rMax = (N - step) / x;
            for (long long r = 1; r <= rMax; ++r) {
                const long long bHi = (N - x * r) / step;
                if (bHi < 1) continue;
                has_updates = true;
                const long long Lbase = x * r;
                if constexpr (C == 0) {
                    diff.add(N, step, Lbase, 1, bHi, 4, 0, 0);
                } else if constexpr (C == 1) {

                    diff.add(N, step, Lbase, 1, bHi,
                             (i128)4 * step * r,
                             (i128)8 * step,
                             0);
                } else {

                    diff.add(N, step, Lbase, 1, bHi,
                             (i128)4 * x * y * r * r,
                             (i128)4 * step * step * r,
                             (i128)4 * step * step);
                }
            }
        }

        if (has_updates) diff.flush(N, step, G);
    }

    return G;
}

template<int C>
static void add_prefixed_component_all_values(
        int N, const vector<int8_t>& mu, vector<i128>& answer) {
    vector<i128> G = build_G_component<C>(N);

    for (int i = 1; i <= N; ++i) {
        G[(size_t)i] += G[(size_t)(i - 1)];
    }

    // For fixed d, G[floor(n/d)] is constant on
    // q*d <= n < (q+1)*d.  Range additions recover the complete transformed
    // prefix in sum_d O(N/d)=O(N log N) time.
    vector<i128> delta((size_t)N + 2, 0);
    for (int d = 1; d <= N; ++d) {
        const int md = (int)mu[(size_t)d];
        if (md == 0) continue;
        i128 factor = (i128)md;
        if constexpr (C >= 1) factor *= d;
        if constexpr (C >= 2) factor *= d;

        const int q_max = N / d;
        for (int q = 1; q <= q_max; ++q) {
            const int left = q * d;
            const int right = std::min(N, left + d - 1);
            const i128 value = factor * G[(size_t)q];
            delta[(size_t)left] += value;
            delta[(size_t)right + 1] -= value;
        }
    }

    i128 current = 0;
    for (int n = 1; n <= N; ++n) {
        current += delta[(size_t)n];
        const i128 nn = (i128)n;
        if constexpr (C == 0) {
            answer[(size_t)n] += nn * nn * current;
        } else if constexpr (C == 1) {
            answer[(size_t)n] -= nn * current;
        } else {
            answer[(size_t)n] += current;
        }
    }
}

static vector<i128> compute_all_F(int N) {
    vector<int8_t> mu = mobius_sieve(N);
    vector<i128> answer((size_t)N + 1, 0);
    for (int n = 1; n <= N; ++n) answer[(size_t)n] = F0_closed((i128)n);
    add_prefixed_component_all_values<0>(N, mu, answer);
    add_prefixed_component_all_values<1>(N, mu, answer);
    add_prefixed_component_all_values<2>(N, mu, answer);
    return answer;
}

static i128 brute_F(int n) {
    i128 ans = F0_closed((i128)n);
    for (int u = 2; u <= n; ++u) {
        for (int v = 1; v < u; ++v) {
            if (std::gcd(u, v) != 1) continue;
            for (int x = 1; x * u <= n; ++x) {
                for (int y = 1; y <= x; ++y) {
                    const int L = x * u + y * v;
                    if (L > n) break;
                    const int K = x * v + y * u;
                    const int mult = (x == y ? 2 : 4);
                    ans += (i128)mult * (n - L) * (n - K);
                }
            }
        }
    }
    return ans;
}

static bool self_test() {
    const vector<i128> fast = compute_all_F(96);
    for (int n = 1; n <= 96; ++n) {
        const i128 slow = brute_F(n);
        if (fast[(size_t)n] != slow) {
            std::cerr << "self-test failed at N=" << n
                      << " fast=" << to_string_i128(fast[(size_t)n])
                      << " brute=" << to_string_i128(slow) << '\n';
            return false;
        }
    }
    std::cerr << "self-test OK for N=1..96\n";
    return true;
}

static bool parse_int(const char* s, int& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    if (v < -2147483647L || v > 2147483647L) return false;
    out = (int)v;
    return true;
}

static bool configure_benchmark_process() {
#ifdef _WIN32
    HANDLE process = GetCurrentProcess();
    if (!SetPriorityClass(process, HIGH_PRIORITY_CLASS)) {
        std::cerr << "failed to set High priority (error "
                  << GetLastError() << ")\n";
        return false;
    }
    if (!SetProcessAffinityMask(process, static_cast<DWORD_PTR>(0x4))) {
        std::cerr << "failed to set affinity mask 0x4 (error "
                  << GetLastError() << ")\n";
        return false;
    }
#else
    std::cerr << "this benchmark requires Windows process controls\n";
    return false;
#endif
    return true;
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc == 2 && std::string(argv[1]) == "--self-test")
        return self_test() ? 0 : 1;

    if (argc != 4) {
        std::cerr << "usage: " << argv[0]
                  << " FIRST_POWER LAST_POWER REPEATS\n"
                  << "       " << argv[0] << " --self-test\n";
        return 2;
    }

    int first_power = -1;
    int last_power = -1;
    int repeats = -1;
    if (!parse_int(argv[1], first_power)
        || !parse_int(argv[2], last_power)
        || !parse_int(argv[3], repeats)
        || first_power < 0 || last_power < first_power
        || last_power > 30 || repeats < 1) {
        std::cerr << "require 0 <= FIRST_POWER <= LAST_POWER <= 30 "
                  << "and REPEATS >= 1\n";
        return 2;
    }

    if (!configure_benchmark_process()) return 1;
    std::cerr << "priority=High affinity_mask=0x4 output_excluded=true\n";
    std::cout << "power\tn\ttrial\tseconds\tresult\n";
    std::cout << std::fixed << std::setprecision(9);

    for (int power = first_power; power <= last_power; ++power) {
        const int N = 1 << power;
        for (int trial = 1; trial <= repeats; ++trial) {
            const auto start = std::chrono::steady_clock::now();
            const vector<i128> answer = compute_all_F(N);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << power << '\t' << N << '\t' << trial << '\t'
                      << seconds << '\t'
                      << to_string_i128(answer[(size_t)N]) << '\n';
            std::cout.flush();
        }
    }
    return 0;
}
