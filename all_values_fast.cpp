#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

// Exact values of OEIS A085582 for 1 <= n <= N.
// Benchmark build: g++ -std=c++20 -O3 -DNDEBUG -march=native
//                  all_values_new.cpp -o all_values_new.exe
// Run: all_values_new.exe FIRST_POWER LAST_POWER REPEATS
// Test: all_values_new.exe --self-test

using std::uint32_t;
using std::uint64_t;
using std::vector;

constexpr int NAIVE_CUTOFF = 48;
constexpr int STAIRCASE_FUSION_DEPTH = 4;
constexpr int MAX_N = 1 << 30;
constexpr uint64_t NTT_MOD_1 = 2305842979148922881ULL;
constexpr uint64_t NTT_MOD_2 = 2305842811645198337ULL;
constexpr uint64_t NTT_ROOT = 3;
static_assert((NTT_MOD_1 - 1) % (uint64_t(1) << 32) == 0);
static_assert((NTT_MOD_2 - 1) % (uint64_t(1) << 32) == 0);

template <class T>
void pointwise_multiply(vector<T> &a, const vector<T> &b) {
    assert(a.size() == b.size());
    auto worker = [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) a[i] *= b[i];
    };
    worker(0, a.size());
}

template <class T>
void pointwise_square(vector<T> &a) {
    auto worker = [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) a[i] *= a[i];
    };
    worker(0, a.size());
}

// F(2^30) has 122 bits, so two limbs suffice for every supported output.
struct BigUInt {
    static constexpr int LIMBS = 2;
    std::array<uint64_t, LIMBS> limb{}; // little endian

    BigUInt(uint64_t x = 0) { limb[0] = x; }

    uint64_t mod_small(uint64_t p) const {
        uint64_t r = 0;
        for (int i = LIMBS - 1; i >= 0; --i)
            r = static_cast<uint64_t>(((__uint128_t(r) << 64) + limb[i]) % p);
        return r;
    }

    void multiply_small(uint64_t x) {
        __uint128_t carry = 0;
        for (uint64_t &v : limb) {
            const __uint128_t z = __uint128_t(v) * x + carry;
            v = static_cast<uint64_t>(z);
            carry = z >> 64;
        }
        if (carry) throw std::overflow_error("BigUInt overflow");
    }

    void add_multiple(const BigUInt &a, uint64_t x) {
        __uint128_t carry = 0;
        for (int i = 0; i < LIMBS; ++i) {
            const __uint128_t z = __uint128_t(a.limb[i]) * x + limb[i] + carry;
            limb[i] = static_cast<uint64_t>(z);
            carry = z >> 64;
        }
        if (carry) throw std::overflow_error("BigUInt overflow");
    }

    friend bool operator<=(const BigUInt &a, const BigUInt &b) {
        for (int i = LIMBS - 1; i >= 0; --i) {
            if (a.limb[i] != b.limb[i]) return a.limb[i] < b.limb[i];
        }
        return true;
    }

    uint32_t divide_small(uint32_t d) {
        __uint128_t r = 0;
        for (int i = LIMBS - 1; i >= 0; --i) {
            __uint128_t cur = (r << 64) + limb[i];
            limb[i] = static_cast<uint64_t>(cur / d);
            r = cur % d;
        }
        return static_cast<uint32_t>(r);
    }

    bool zero() const {
        for (uint64_t x : limb) if (x != 0) return false;
        return true;
    }

    std::string str() const {
        if (zero()) return "0";
        BigUInt x = *this;
        vector<uint32_t> part;
        while (!x.zero()) part.push_back(x.divide_small(1000000000));
        std::ostringstream out;
        out << part.back();
        for (int i = static_cast<int>(part.size()) - 2; i >= 0; --i)
            out << std::setw(9) << std::setfill('0') << part[i];
        return out.str();
    }
};

template <uint64_t MOD> struct Mint {
    static_assert((MOD & 1) == 1 && MOD < (1ULL << 62), "Montgomery modulus out of range");
    uint64_t v;

    static constexpr uint64_t inverse_mod_2_64() {
        uint64_t x = MOD;
        for (int i = 0; i < 6; ++i) x *= 2 - MOD * x;
        return x;
    }
    static constexpr uint64_t NEG_INV = 0 - inverse_mod_2_64();
    static constexpr uint64_t R_MOD = static_cast<uint64_t>((__uint128_t(1) << 64) % MOD);
    static constexpr uint64_t R2 = static_cast<uint64_t>((__uint128_t(R_MOD) * R_MOD) % MOD);

    static uint64_t reduce_lazy(__uint128_t z) {
        const uint64_t q = static_cast<uint64_t>(z) * NEG_INV;
        const __uint128_t qm = __uint128_t(q) * MOD;
        const uint64_t zlo = static_cast<uint64_t>(z);
        const uint64_t qlo = static_cast<uint64_t>(qm);
        const uint64_t carry = (zlo + qlo < zlo);
        uint64_t hi = static_cast<uint64_t>(z >> 64) + static_cast<uint64_t>(qm >> 64) + carry;
        return hi;
    }
    static uint64_t reduce(__uint128_t z) {
        uint64_t hi = reduce_lazy(z);
        if (hi >= MOD) hi -= MOD;
        return hi;
    }

    Mint(long long x = 0) {
        uint64_t y;
        if (x >= 0 && static_cast<uint64_t>(x) < MOD) {
            y = static_cast<uint64_t>(x);
        } else {
            long long r = x % static_cast<long long>(MOD);
            if (r < 0) r += static_cast<long long>(MOD);
            y = static_cast<uint64_t>(r);
        }
        v = reduce(__uint128_t(y) * R2);
    }
    static Mint raw(uint64_t x) { Mint a; a.v = x; return a; }
    static Mint from_u64(uint64_t x) {
        return raw(reduce(__uint128_t(x % MOD) * R2));
    }
    uint64_t value() const { return reduce(v); }
    Mint &operator+=(Mint b) {
        uint64_t x = v + b.v;
        if (x >= MOD) x -= MOD;
        v = x;
        return *this;
    }
    Mint &operator-=(Mint b) {
        v = v >= b.v ? v - b.v : v + MOD - b.v;
        return *this;
    }
    Mint &operator*=(Mint b) {
        v = reduce(__uint128_t(v) * b.v);
        return *this;
    }
    // Harvey-style redundant representatives used only inside NTT.  Inputs
    // are in [0,2p); add/sub return [0,2p), while unreduced sums/differences
    // are in [0,4p) and must be consumed immediately by multiplication.
    static Mint add_lazy(Mint a, Mint b) {
        uint64_t x = a.v + b.v;
        if (x >= 2 * MOD) x -= 2 * MOD;
        return raw(x);
    }
    static Mint sub_lazy(Mint a, Mint b) {
        uint64_t x = a.v + 2 * MOD - b.v;
        if (x >= 2 * MOD) x -= 2 * MOD;
        return raw(x);
    }
    static Mint add_unreduced(Mint a, Mint b) { return raw(a.v + b.v); }
    static Mint sub_unreduced(Mint a, Mint b) { return raw(a.v + 2 * MOD - b.v); }
    static Mint mul_lazy(Mint a, Mint b) {
        return raw(reduce_lazy(__uint128_t(a.v) * b.v));
    }
    friend Mint operator+(Mint a, Mint b) { return a += b; }
    friend Mint operator-(Mint a, Mint b) { return a -= b; }
    friend Mint operator*(Mint a, Mint b) { return a *= b; }
    Mint operator-() const { return v ? raw(MOD - v) : raw(0); }
    bool operator==(Mint b) const { return v == b.v; }
    bool operator!=(Mint b) const { return v != b.v; }
    static Mint power(Mint a, uint64_t e) {
        Mint r = 1;
        while (e) {
            if (e & 1) r *= a;
            a *= a;
            e >>= 1;
        }
        return r;
    }
    Mint inv() const { return power(*this, MOD - 2); }
};

template <uint64_t MOD, uint64_t ROOT> struct NTT {
    using M = Mint<MOD>;

    static constexpr int two_adic_rank() {
        uint64_t x = MOD - 1;
        int rank = 0;
        while ((x & 1) == 0) {
            ++rank;
            x >>= 1;
        }
        return rank;
    }

    static constexpr int RANK = two_adic_rank();

    struct Info {
        std::array<M, RANK + 1> root{}, iroot{}, inv_size{}, rate2{}, irate2{}, rate3{}, irate3{};

        Info() {
            root[RANK] = M::power(M::from_u64(ROOT), (MOD - 1) >> RANK);
            iroot[RANK] = root[RANK].inv();
            for (int i = RANK - 1; i >= 0; --i) {
                root[i] = root[i + 1] * root[i + 1];
                iroot[i] = iroot[i + 1] * iroot[i + 1];
            }
            inv_size[0] = 1;
            const M inv2 = M(2).inv();
            for (int i = 1; i <= RANK; ++i) inv_size[i] = inv_size[i - 1] * inv2;
            M prod = 1, iprod = 1;
            for (int i = 0; i <= RANK - 2; ++i) {
                rate2[i] = root[i + 2] * prod;
                irate2[i] = iroot[i + 2] * iprod;
                prod *= iroot[i + 2];
                iprod *= root[i + 2];
            }
            prod = 1;
            iprod = 1;
            for (int i = 0; i <= RANK - 3; ++i) {
                rate3[i] = root[i + 3] * prod;
                irate3[i] = iroot[i + 3] * iprod;
                prod *= iroot[i + 3];
                iprod *= root[i + 3];
            }
        }
    };

    static const Info &info() {
        static const Info value;
        return value;
    }

    static int log2_exact(size_t n) {
        int h = 0;
        while ((size_t(1) << h) < n) ++h;
        return h;
    }

    static int trailing_ones_index(size_t s) {
        return __builtin_ctzll(~static_cast<unsigned long long>(s));
    }

    static M rotation_for_block(size_t block, const Info &f,
                                int first_root, bool inverse) {
        M rotation = 1;
        int root_index = first_root;
        while (block) {
            if (block & 1)
                rotation *= inverse ? f.iroot[root_index] : f.root[root_index];
            block >>= 1;
            ++root_index;
        }
        return rotation;
    }

    static void forward(vector<M> &a) {
        const int h = log2_exact(a.size());
        const Info &f = info();
        int len = 0;
        while (len < h) {
            if (h - len == 1) {
                const size_t p = size_t(1) << (h - len - 1);
                const size_t blocks = size_t(1) << len;
                auto worker = [&](size_t begin, size_t end) {
                    size_t position = begin;
                    size_t s = position / p;
                    M rot = rotation_for_block(s, f, 2, false);
                    while (position < end) {
                        const size_t block_begin = s * p;
                        const size_t i_begin = position - block_begin;
                        const size_t i_end = std::min(p, i_begin + end - position);
                        const size_t offset = s << (h - len);
                        if (s == 0) {
                            for (size_t i = i_begin; i < i_end; ++i) {
                                const M l = a[offset + i];
                                const M r = a[offset + i + p];
                                a[offset + i] = M::add_lazy(l, r);
                                a[offset + i + p] = M::sub_lazy(l, r);
                            }
                        } else {
                            for (size_t i = i_begin; i < i_end; ++i) {
                                const M l = a[offset + i];
                                const M r = M::mul_lazy(a[offset + i + p], rot);
                                a[offset + i] = M::add_lazy(l, r);
                                a[offset + i + p] = M::sub_lazy(l, r);
                            }
                        }
                        position += i_end - i_begin;
                        if (i_end == p && position < end) {
                            rot *= f.rate2[trailing_ones_index(s)];
                            ++s;
                        }
                    }
                };
                worker(0, blocks * p);
                ++len;
            } else {
                const size_t p = size_t(1) << (h - len - 2);
                const M imag = f.root[2];
                const size_t blocks = size_t(1) << len;
                auto worker = [&](size_t begin, size_t end) {
                    size_t position = begin;
                    size_t s = position / p;
                    M rot = rotation_for_block(s, f, 3, false);
                    while (position < end) {
                        const size_t block_begin = s * p;
                        const size_t i_begin = position - block_begin;
                        const size_t i_end = std::min(p, i_begin + end - position);
                        const M rot2 = rot * rot;
                        const M rot3 = rot2 * rot;
                        const size_t offset = s << (h - len);
                        if (s == 0) {
                            for (size_t i = i_begin; i < i_end; ++i) {
                                const M a0 = a[offset + i];
                                const M a1 = a[offset + i + p];
                                const M a2 = a[offset + i + 2 * p];
                                const M a3 = a[offset + i + 3 * p];
                                const M t0 = M::add_lazy(a0, a2);
                                const M t1 = M::sub_lazy(a0, a2);
                                const M t2 = M::add_lazy(a1, a3);
                                const M t3 = M::mul_lazy(M::sub_unreduced(a1, a3), imag);
                                a[offset + i] = M::add_lazy(t0, t2);
                                a[offset + i + p] = M::sub_lazy(t0, t2);
                                a[offset + i + 2 * p] = M::add_lazy(t1, t3);
                                a[offset + i + 3 * p] = M::sub_lazy(t1, t3);
                            }
                        } else {
                            for (size_t i = i_begin; i < i_end; ++i) {
                                const M a0 = a[offset + i];
                                const M a1 = M::mul_lazy(a[offset + i + p], rot);
                                const M a2 = M::mul_lazy(a[offset + i + 2 * p], rot2);
                                const M a3 = M::mul_lazy(a[offset + i + 3 * p], rot3);
                                const M t0 = M::add_lazy(a0, a2);
                                const M t1 = M::sub_lazy(a0, a2);
                                const M t2 = M::add_lazy(a1, a3);
                                const M t3 = M::mul_lazy(M::sub_unreduced(a1, a3), imag);
                                a[offset + i] = M::add_lazy(t0, t2);
                                a[offset + i + p] = M::sub_lazy(t0, t2);
                                a[offset + i + 2 * p] = M::add_lazy(t1, t3);
                                a[offset + i + 3 * p] = M::sub_lazy(t1, t3);
                            }
                        }
                        position += i_end - i_begin;
                        if (i_end == p && position < end) {
                            rot *= f.rate3[trailing_ones_index(s)];
                            ++s;
                        }
                    }
                };
                worker(0, blocks * p);
                len += 2;
            }
        }
    }

    static void inverse(vector<M> &a) {
        const int h = log2_exact(a.size());
        const Info &f = info();
        const M inv_n = f.inv_size[h];
        int len = h;
        while (len > 0) {
            if (len == 1) {
                const size_t p = size_t(1) << (h - len);
                auto worker = [&](size_t begin, size_t end) {
                    for (size_t i = begin; i < end; ++i) {
                        const M l = a[i];
                        const M r = a[i + p];
                        // Fold the mandatory multiplication by n^{-1} into
                        // the last butterfly.  The unreduced inputs are below
                        // 4p and are normalized by the Montgomery product.
                        a[i] = M::add_unreduced(l, r) * inv_n;
                        a[i + p] = M::sub_unreduced(l, r) * inv_n;
                    }
                };
                worker(0, p);
                --len;
            } else {
                const size_t p = size_t(1) << (h - len);
                const M iimag = f.iroot[2];
                const size_t blocks = size_t(1) << (len - 2);
                auto worker = [&](size_t begin, size_t end) {
                    size_t position = begin;
                    size_t s = position / p;
                    M irot = rotation_for_block(s, f, 3, true);
                    while (position < end) {
                        const size_t block_begin = s * p;
                        const size_t i_begin = position - block_begin;
                        const size_t i_end = std::min(p, i_begin + end - position);
                        const M irot2 = irot * irot;
                        const M irot3 = irot2 * irot;
                        const size_t offset = s << (h - len + 2);
                        if (s == 0) {
                            for (size_t i = i_begin; i < i_end; ++i) {
                                const M a0 = a[offset + i];
                                const M a1 = a[offset + i + p];
                                const M a2 = a[offset + i + 2 * p];
                                const M a3 = a[offset + i + 3 * p];
                                const M t0 = M::add_lazy(a0, a1);
                                const M t1 = M::sub_lazy(a0, a1);
                                const M t2 = M::add_lazy(a2, a3);
                                const M t3 = M::mul_lazy(M::sub_unreduced(a2, a3), iimag);
                                if (len == 2) {
                                    a[offset + i] = M::add_unreduced(t0, t2) * inv_n;
                                    a[offset + i + p] = M::add_unreduced(t1, t3) * inv_n;
                                    a[offset + i + 2 * p] = M::sub_unreduced(t0, t2) * inv_n;
                                    a[offset + i + 3 * p] = M::sub_unreduced(t1, t3) * inv_n;
                                } else {
                                    a[offset + i] = M::add_lazy(t0, t2);
                                    a[offset + i + p] = M::add_lazy(t1, t3);
                                    a[offset + i + 2 * p] = M::sub_lazy(t0, t2);
                                    a[offset + i + 3 * p] = M::sub_lazy(t1, t3);
                                }
                            }
                        } else {
                            for (size_t i = i_begin; i < i_end; ++i) {
                                const M a0 = a[offset + i];
                                const M a1 = a[offset + i + p];
                                const M a2 = a[offset + i + 2 * p];
                                const M a3 = a[offset + i + 3 * p];
                                const M t0 = M::add_lazy(a0, a1);
                                const M t1 = M::sub_lazy(a0, a1);
                                const M t2 = M::add_lazy(a2, a3);
                                const M t3 = M::mul_lazy(M::sub_unreduced(a2, a3), iimag);
                                a[offset + i] = M::add_lazy(t0, t2);
                                a[offset + i + p] =
                                    M::mul_lazy(M::add_unreduced(t1, t3), irot);
                                a[offset + i + 2 * p] =
                                    M::mul_lazy(M::sub_unreduced(t0, t2), irot2);
                                a[offset + i + 3 * p] =
                                    M::mul_lazy(M::sub_unreduced(t1, t3), irot3);
                            }
                        }
                        position += i_end - i_begin;
                        if (i_end == p && position < end) {
                            irot *= f.irate3[trailing_ones_index(s)];
                            ++s;
                        }
                    }
                };
                worker(0, blocks * p);
                len -= 2;
            }
        }
    }

    static void transform(vector<M> &a, bool invert) {
        if (invert) {
            inverse(a);
        } else {
            forward(a);
        }
    }

};

// Correct convolution wrapper preserving input lengths.  Kept outside NTT to
// make truncation explicit and avoid accidental dependence on trailing zeros.
template <uint64_t MOD, uint64_t ROOT>
vector<Mint<MOD>> convolve(vector<Mint<MOD>> a, vector<Mint<MOD>> b) {
    using M = Mint<MOD>;
    if (a.empty() || b.empty()) return {};
    const size_t need = a.size() + b.size() - 1;
    if (std::min(a.size(), b.size()) <= NAIVE_CUTOFF) {
        vector<M> c(need);
        for (size_t i = 0; i < a.size(); ++i)
            for (size_t j = 0; j < b.size(); ++j)
                c[i + j] += a[i] * b[j];
        return c;
    }
    size_t n = 1;
    while (n < need) n <<= 1;
    if ((MOD - 1) % n != 0) throw std::runtime_error("convolution exceeds modulus root order");
    a.resize(n);
    b.resize(n);
    NTT<MOD, ROOT>::transform(a, false);
    NTT<MOD, ROOT>::transform(b, false);
    pointwise_multiply(a, b);
    NTT<MOD, ROOT>::transform(a, true);
    a.resize(need);
    return a;
}

// A square needs only one forward transform.  This is used when a tree for
// denominator power two is obtained from the already available power-one
// tree, node by node.
template <uint64_t MOD, uint64_t ROOT>
vector<Mint<MOD>> convolve_square(vector<Mint<MOD>> a) {
    using M = Mint<MOD>;
    if (a.empty()) return {};
    const size_t need = 2 * a.size() - 1;
    if (a.size() <= NAIVE_CUTOFF) {
        vector<M> c(need);
        for (size_t i = 0; i < a.size(); ++i)
            for (size_t j = 0; j < a.size(); ++j)
                c[i + j] += a[i] * a[j];
        return c;
    }
    size_t n = 1;
    while (n < need) n <<= 1;
    if ((MOD - 1) % n != 0) throw std::runtime_error("square exceeds modulus root order");
    a.resize(n);
    NTT<MOD, ROOT>::transform(a, false);
    pointwise_square(a);
    NTT<MOD, ROOT>::transform(a, true);
    a.resize(need);
    return a;
}

// Return only coefficients [0, prefix).
template <uint64_t MOD, uint64_t ROOT>
vector<Mint<MOD>> convolve_prefix(vector<Mint<MOD>> a,
                                  vector<Mint<MOD>> b, size_t prefix) {
    if (a.empty() || b.empty() || prefix == 0) return {};
    if (a.size() > prefix) a.resize(prefix);
    if (b.size() > prefix) b.resize(prefix);
    vector<Mint<MOD>> result = convolve<MOD, ROOT>(std::move(a), std::move(b));
    if (result.size() > prefix) result.resize(prefix);
    return result;
}

template <uint64_t MOD, uint64_t ROOT>
vector<Mint<MOD>> convolve_square_prefix(vector<Mint<MOD>> a, size_t prefix) {
    if (a.empty() || prefix == 0) return {};
    if (a.size() > prefix) a.resize(prefix);
    vector<Mint<MOD>> result = convolve_square<MOD, ROOT>(std::move(a));
    if (result.size() > prefix) result.resize(prefix);
    return result;
}

template <uint64_t MOD, uint64_t ROOT> class Solver {
    using M = Mint<MOD>;

    struct Poly {
        int shift = 0;
        vector<M> a;
        bool zero() const { return a.empty(); }
    };

    struct Spec {
        int lo, hi;
        int active_lo;
        int denominator_power;
        int slope;
        long long scale;
        bool multiply_by_index;
        // A second monomial is used only when its slope is adjacent to the
        // first one.  Keeping the two monomials in one local numerator then
        // preserves the O(K|I|) dense span while allowing two rank-one
        // staircase terms to be combined before the recursion.
        int second_slope = 0;
        long long second_scale = 0;
        int second_active_lo = -1;
    };

    struct Node {
        int lo, hi;
        Poly D, P, P2;
        const Node *denominator_alias = nullptr;
        std::unique_ptr<Node> left, right;
    };

    int N, K;
    const vector<int> &primes;

    static void trim(Poly &p) {
        while (!p.a.empty() && p.a.back() == M(0)) p.a.pop_back();
        if (p.a.empty()) p.shift = 0;
    }

    static const Node *denominator_owner(const Node *node) {
        while (node->denominator_alias) node = node->denominator_alias;
        return node;
    }

    static const Poly &denominator_poly(const Node *node) {
        return denominator_owner(node)->D;
    }

    Poly scalar(Poly p, M c) const {
        if (c == M(0)) return {};
        if (c == M(1)) return p;
        if (c == M(-1)) {
            for (M &x : p.a) x = -x;
            return p;
        }
        if (c == M(2)) {
            for (M &x : p.a) x += x;
            return p;
        }
        for (M &x : p.a) x *= c;
        trim(p);
        return p;
    }

    Poly add(const Poly &x, const Poly &y) const {
        if (x.zero()) return y;
        if (y.zero()) return x;
        int s = std::min(x.shift, y.shift);
        const long long x_end = static_cast<long long>(x.shift) + static_cast<long long>(x.a.size());
        const long long y_end = static_cast<long long>(y.shift) + static_cast<long long>(y.a.size());
        int e = static_cast<int>(std::min<long long>(N + 1LL, std::max(x_end, y_end)));
        if (s >= e) return {};
        Poly z;
        z.shift = s;
        z.a.assign(e - s, M(0));
        for (int i = 0; i < static_cast<int>(x.a.size()) && x.shift + i < e; ++i)
            z.a[x.shift + i - s] += x.a[i];
        for (int i = 0; i < static_cast<int>(y.a.size()) && y.shift + i < e; ++i)
            z.a[y.shift + i - s] += y.a[i];
        trim(z);
        return z;
    }

    Poly sub(const Poly &x, const Poly &y) const { return add(x, scalar(y, M(-1))); }

    // Formal division by 1-x modulo x^(N+1).  Every use below is known
    // algebraically to contain this factor; the prefix sum also remains
    // valid when the stored polynomial has a positive shift.
    Poly divide_by_one_minus_x(Poly p) const {
        M prefix = 0;
        for (M &coefficient : p.a) {
            prefix += coefficient;
            coefficient = prefix;
        }
        trim(p);
        return p;
    }

    Poly divide_by_one_minus_q_power(Poly p, int power) const {
        if (p.zero()) return {};
        p.a.resize(static_cast<size_t>(N - p.shift + 1), M(0));
        for (size_t i = static_cast<size_t>(power); i < p.a.size(); ++i)
            p.a[i] += p.a[i - power];
        trim(p);
        return p;
    }

    Poly shift_by(Poly p, int amount) const {
        if (p.zero() || static_cast<long long>(p.shift) + amount > N) return {};
        p.shift += amount;
        const size_t limit = static_cast<size_t>(N - p.shift + 1);
        if (p.a.size() > limit) p.a.resize(limit);
        trim(p);
        return p;
    }

    Poly mul(const Poly &x, const Poly &y) const {
        const long long shift_sum = static_cast<long long>(x.shift) + y.shift;
        if (x.zero() || y.zero() || shift_sum > N) return {};
        Poly z;
        z.shift = static_cast<int>(shift_sum);
        const int limit = N - z.shift + 1;
        vector<M> a = x.a, b = y.a;
        if (static_cast<int>(a.size()) > limit) a.resize(limit);
        if (static_cast<int>(b.size()) > limit) b.resize(limit);
        z.a = convolve_prefix<MOD, ROOT>(std::move(a), std::move(b), limit);
        trim(z);
        return z;
    }

    // Sum two aligned products in the frequency domain, replacing two
    // inverse transforms by one whenever their transform lengths agree.
    Poly mul_pair_sum(const Poly &a, const Poly &b, M factor1,
                      const Poly &c, const Poly &d, M factor2) const {
        const auto fallback = [&] {
            return add(scalar(mul(a, b), factor1),
                       scalar(mul(c, d), factor2));
        };
        if (a.zero() || b.zero() || c.zero() || d.zero()) return fallback();

        const long long shift1 = static_cast<long long>(a.shift) + b.shift;
        const long long shift2 = static_cast<long long>(c.shift) + d.shift;
        if (shift1 != shift2 || shift1 > N) return fallback();
        const int shift = static_cast<int>(shift1);
        const size_t limit = static_cast<size_t>(N - shift + 1);
        const size_t as = std::min(a.a.size(), limit);
        const size_t bs = std::min(b.a.size(), limit);
        const size_t cs = std::min(c.a.size(), limit);
        const size_t ds = std::min(d.a.size(), limit);
        if (std::min(as, bs) <= NAIVE_CUTOFF ||
            std::min(cs, ds) <= NAIVE_CUTOFF)
            return fallback();

        const size_t need1 = as + bs - 1;
        const size_t need2 = cs + ds - 1;
        size_t n1 = 1, n2 = 1;
        while (n1 < need1) n1 <<= 1;
        while (n2 < need2) n2 <<= 1;
        const size_t n = std::max(n1, n2);
        if ((MOD - 1) % n != 0) return fallback();

        auto copy_scaled = [](const vector<M> &source, size_t count,
                              vector<M> &target, M factor) {
            if (factor == M(1)) {
                std::copy(source.begin(), source.begin() + count, target.begin());
            } else if (factor == M(-1)) {
                for (size_t i = 0; i < count; ++i) target[i] = -source[i];
            } else if (factor == M(2)) {
                for (size_t i = 0; i < count; ++i)
                    target[i] = source[i] + source[i];
            } else {
                for (size_t i = 0; i < count; ++i) target[i] = source[i] * factor;
            }
        };

        vector<M> accumulator(n), temporary(n);
        copy_scaled(a.a, as, accumulator, factor1);
        std::copy(b.a.begin(), b.a.begin() + bs, temporary.begin());
        NTT<MOD, ROOT>::transform(accumulator, false);
        NTT<MOD, ROOT>::transform(temporary, false);
        pointwise_multiply(accumulator, temporary);

        vector<M> right(n);
        std::fill(temporary.begin(), temporary.end(), M(0));
        copy_scaled(c.a, cs, temporary, factor2);
        std::copy(d.a.begin(), d.a.begin() + ds, right.begin());
        NTT<MOD, ROOT>::transform(temporary, false);
        NTT<MOD, ROOT>::transform(right, false);
        auto accumulate = [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
                accumulator[i] += temporary[i] * right[i];
        };
        accumulate(0, n);
        vector<M>().swap(temporary);
        vector<M>().swap(right);

        NTT<MOD, ROOT>::transform(accumulator, true);
        if (accumulator.size() > limit) accumulator.resize(limit);
        Poly result{shift, std::move(accumulator)};
        trim(result);
        return result;
    }

    // Two quotients by the same root denominator need the same transformed
    // inverse series.  Keep that spectrum immutable and process the two
    // numerators sequentially, saving one forward NTT without increasing the
    // number of simultaneously live transform-sized work arrays.
    std::pair<Poly, Poly> mul_two_shared_right(const Poly &a, const Poly &c,
                                               const Poly &b) const {
        const auto fallback = [&] {
            return std::make_pair(mul(a, b), mul(c, b));
        };
        if (a.zero() || c.zero() || b.zero() || b.shift != 0 ||
            a.shift != c.shift || a.shift > N)
            return fallback();

        const int shift = a.shift;
        const size_t limit = static_cast<size_t>(N - shift + 1);
        const size_t as = std::min(a.a.size(), limit);
        const size_t cs = std::min(c.a.size(), limit);
        const size_t bs = std::min(b.a.size(), limit);
        if (std::min(as, bs) <= NAIVE_CUTOFF ||
            std::min(cs, bs) <= NAIVE_CUTOFF)
            return fallback();

        const auto plan = [&](size_t left_size) {
            const size_t need = left_size + bs - 1;
            size_t n = 1;
            while (n < need) n <<= 1;
            return n;
        };
        const size_t first_n = plan(as);
        const size_t second_n = plan(cs);
        if (first_n != second_n || (MOD - 1) % first_n != 0)
            return fallback();

        vector<M> right(b.a.begin(), b.a.begin() + bs);
        right.resize(first_n);
        NTT<MOD, ROOT>::transform(right, false);

        const auto multiply_one = [&](const Poly &left, size_t left_size,
                                      size_t transform_size) {
            vector<M> product(left.a.begin(), left.a.begin() + left_size);
            product.resize(transform_size);
            NTT<MOD, ROOT>::transform(product, false);
            pointwise_multiply(product, right);
            NTT<MOD, ROOT>::transform(product, true);
            if (product.size() > limit) product.resize(limit);
            Poly result{shift, std::move(product)};
            trim(result);
            return result;
        };

        Poly first = multiply_one(a, as, first_n);
        Poly second = multiply_one(c, cs, second_n);
        return {std::move(first), std::move(second)};
    }

    Poly square(const Poly &x) const {
        if (x.zero() || 2LL * x.shift > N) return {};
        Poly z;
        z.shift = 2 * x.shift;
        const int limit = N - z.shift + 1;
        vector<M> a = x.a;
        if (static_cast<int>(a.size()) > limit) a.resize(limit);
        z.a = convolve_square_prefix<MOD, ROOT>(std::move(a), limit);
        trim(z);
        return z;
    }

    Poly mul_by_denominator(const Poly &x, const Node *denominator) const {
        return mul(x, denominator_poly(denominator));
    }

    // Compute x*Dx + y*Dy with one inverse transform.  Dx and Dy are fixed
    // denominator polynomials.  Separate convolutions would transform both
    // inputs and invert both products; after aligning the two shifts we may
    // add the products in frequency space and invert only their sum.  This is
    // the binary building block of FFT trading.
    Poly lift_pair(const Poly &x, const Node *dx,
                   const Poly &y, const Node *dy) const {
        if (x.zero()) return mul_by_denominator(y, dy);
        if (y.zero()) return mul_by_denominator(x, dx);

        const Poly &fx = denominator_poly(dx);
        const Poly &fy = denominator_poly(dy);
        if (fx.shift != 0 || fy.shift != 0)
            throw std::logic_error("lift denominator has a shift");

        const int shift = std::min(x.shift, y.shift);
        if (shift > N) return {};
        const size_t limit = static_cast<size_t>(N - shift + 1);
        const size_t ox = static_cast<size_t>(x.shift - shift);
        const size_t oy = static_cast<size_t>(y.shift - shift);
        if (ox >= limit) return mul_by_denominator(y, dy);
        if (oy >= limit) return mul_by_denominator(x, dx);

        const size_t ax_size = std::min(x.a.size(), limit - ox);
        const size_t ay_size = std::min(y.a.size(), limit - oy);
        const size_t dx_size = std::min(fx.a.size(), limit - ox);
        const size_t dy_size = std::min(fy.a.size(), limit - oy);
        if (std::min(ax_size, dx_size) <= NAIVE_CUTOFF ||
            std::min(ay_size, dy_size) <= NAIVE_CUTOFF)
            return add(mul_by_denominator(x, dx),
                       mul_by_denominator(y, dy));

        const size_t need_x = ax_size + dx_size - 1;
        const size_t need_y = ay_size + dy_size - 1;
        size_t nx = 1, ny = 1, n = 1;
        while (nx < need_x) nx <<= 1;
        while (ny < need_y) ny <<= 1;
        const size_t need = std::max(ox + need_x, oy + need_y);
        while (n < need) n <<= 1;

        // Shift alignment may cross a power-of-two boundary.  In that case
        // the saved inverse is not worth doubling every transform.
        if (n > std::max(nx, ny) || (MOD - 1) % n != 0)
            return add(mul_by_denominator(x, dx),
                       mul_by_denominator(y, dy));

        vector<M> tx(n), ty(n);
        std::copy(x.a.begin(), x.a.begin() + ax_size, tx.begin() + ox);
        std::copy(y.a.begin(), y.a.begin() + ay_size, ty.begin() + oy);
        NTT<MOD, ROOT>::transform(tx, false);
        NTT<MOD, ROOT>::transform(ty, false);

        vector<M> sx(fx.a.begin(), fx.a.begin() + dx_size);
        vector<M> sy(fy.a.begin(), fy.a.begin() + dy_size);
        sx.resize(n);
        sy.resize(n);
        NTT<MOD, ROOT>::transform(sx, false);
        NTT<MOD, ROOT>::transform(sy, false);
        auto mix = [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
                tx[i] = tx[i] * sx[i] + ty[i] * sy[i];
        };
        mix(0, n);
        vector<M>().swap(ty);
        vector<M>().swap(sx);
        vector<M>().swap(sy);
        NTT<MOD, ROOT>::transform(tx, true);
        if (tx.size() > limit) tx.resize(limit);

        Poly z{shift, std::move(tx)};
        trim(z);
        return z;
    }

    Poly singleton_denominator(int i, int power) const {
        Poly p;
        p.a.assign(std::min(N, power * i) + 1, M(0));
        p.a[0] = 1;
        if (i <= N) p.a[i] -= M(power);
        if (power == 2 && 2 * i <= N) p.a[2 * i] += M(1);
        trim(p);
        return p;
    }

    std::unique_ptr<Node> build_denominator(int lo, int hi, int power) const {
        auto node = std::make_unique<Node>();
        node->lo = lo;
        node->hi = hi;
        if (lo == hi) {
            node->D = singleton_denominator(lo, power);
            return node;
        }
        int mid = (lo + hi) >> 1;
        std::unique_ptr<Node> left;
        left = build_denominator(lo, mid, power);
        auto right = build_denominator(mid + 1, hi, power);
        node->left = std::move(left);
        node->right = std::move(right);
        node->D = mul(denominator_poly(node->left.get()),
                      denominator_poly(node->right.get()));
        return node;
    }

    std::unique_ptr<Node> alias_denominator_tree(const Node *base) const {
        auto node = std::make_unique<Node>();
        node->lo = base->lo;
        node->hi = base->hi;
        node->denominator_alias = denominator_owner(base);
        if (base->left) {
            node->left = alias_denominator_tree(base->left.get());
            node->right = alias_denominator_tree(base->right.get());
        }
        return node;
    }

    void retarget_denominator_alias(Node *node, const Node *base) const {
        if (node->lo != base->lo || node->hi != base->hi ||
            static_cast<bool>(node->left) != static_cast<bool>(base->left))
            throw std::logic_error("incompatible denominator alias trees");
        node->D = {};
        node->P = {};
        node->P2 = {};
        node->denominator_alias = denominator_owner(base);
        if (node->left) {
            retarget_denominator_alias(node->left.get(), base->left.get());
            retarget_denominator_alias(node->right.get(), base->right.get());
        }
    }

    std::unique_ptr<Node> build_squared_denominator(const Node *base) const {
        auto node = std::make_unique<Node>();
        node->lo = base->lo;
        node->hi = base->hi;
        if (base->left) {
            std::unique_ptr<Node> left;
            left = build_squared_denominator(base->left.get());
            auto right = build_squared_denominator(base->right.get());
            node->left = std::move(left);
            node->right = std::move(right);
        }
        node->D = square(denominator_poly(base));
        return node;
    }

    Poly leaf_numerator(int index, const Spec &s) const {
        const long long index_factor = s.multiply_by_index ? index : 1LL;
        const long long degree1 = 1LL * s.slope * index;
        const long long degree2 = 1LL * s.second_slope * index;
        const int active2 = s.second_active_lo < 0 ? s.active_lo : s.second_active_lo;
        const bool use1 = index >= s.active_lo && degree1 <= N && s.scale != 0;
        const bool use2 = index >= active2 && s.second_slope != 0 &&
                          degree2 <= N && s.second_scale != 0;
        if (!use1 && !use2) return {};

        const long long first = use1 ? degree1 : degree2;
        const long long last = use2 ? std::max(first, degree2) : first;
        const long long begin = use1 && use2 ? std::min(degree1, degree2) : first;
        Poly p;
        p.shift = static_cast<int>(begin);
        p.a.assign(static_cast<size_t>(last - begin + 1), M(0));
        if (use1) p.a[static_cast<size_t>(degree1 - begin)] += M(s.scale * index_factor);
        if (use2) p.a[static_cast<size_t>(degree2 - begin)] += M(s.second_scale * index_factor);
        trim(p);
        return p;
    }

    void fill_numerator(Node *node, const Spec &s) const {
        if (node->lo == node->hi) {
            node->P = leaf_numerator(node->lo, s);
            return;
        }
        Node *left = node->left.get();
        const Spec left_spec = s;
        fill_numerator(left, left_spec);
        fill_numerator(node->right.get(), s);
        node->P = lift_pair(node->left->P, node->right.get(),
                            node->right->P, node->left.get());
    }

    void fill_second_numerator(Node *node, const Spec &s) const {
        if (node->lo == node->hi) {
            node->P2 = leaf_numerator(node->lo, s);
            return;
        }
        Node *left = node->left.get();
        const Spec left_spec = s;
        fill_second_numerator(left, left_spec);
        fill_second_numerator(node->right.get(), s);
        node->P2 = lift_pair(node->left->P2, node->right.get(),
                             node->right->P2, node->left.get());
    }

    // For E_I=prod_{i in I}(1-q^i), logarithmic differentiation gives
    //   sum_{i in I} i*q^i/(1-q^i) = -q*E_I'(q)/E_I(q).
    // Hence the marked numerator for the indexed A-family is obtained from
    // the already built denominator in linear time, with no product-tree
    // convolutions.  Cross recursion needs this identity at every node.
    void fill_indexed_A_from_denominator(Node *node) const {
        if (node->left) {
            fill_indexed_A_from_denominator(node->left.get());
            fill_indexed_A_from_denominator(node->right.get());
        }
        const Poly &d = denominator_poly(node);
        size_t first = 1;
        while (first < d.a.size() && d.a[first] == M(0)) ++first;
        if (first == d.a.size()) {
            node->P2 = {};
            return;
        }
        Poly numerator;
        numerator.shift = static_cast<int>(first);
        numerator.a.resize(d.a.size() - first);
        for (size_t degree = first; degree < d.a.size(); ++degree)
            numerator.a[degree - first] =
                -M(static_cast<long long>(degree)) * d.a[degree];
        trim(numerator);
        node->P2 = std::move(numerator);
    }

    // Numerator of one atom relative to the denominator at an ancestor node.
    // Only one root-to-leaf path is visited, so this costs one geometrically
    // growing polynomial lift per tree level, i.e. O(M(N)) in total.
    Poly lift_single_leaf(const Node *node, int index, const Spec &s) const {
        if (index < node->lo || index > node->hi) return {};
        if (!node->left) return leaf_numerator(index, s);
        if (index <= node->left->hi)
            return mul_by_denominator(lift_single_leaf(node->left.get(), index, s),
                                      node->right.get());
        return mul_by_denominator(lift_single_leaf(node->right.get(), index, s),
                                  node->left.get());
    }

    enum class Kind { Wedge, Triangle };

    static int classify(Kind kind, const Node *x, const Node *y) {
        int a = x->lo, b = x->hi, c = y->lo, d = y->hi;
        if (kind == Kind::Wedge) {
            if (b < c && d < 2 * a) return 1;
            if (a >= d || c >= 2 * b) return 0;
            return -1;
        }
        if (a > d) return 1;
        if (b <= c) return 0;
        return -1;
    }

    Poly cross(const Node *x, const Node *y, Kind kind) const {
        int state = classify(kind, x, y);
        if (state == 1) return mul(x->P, y->P);
        if (state == 0) return {};
        int len_x = x->hi - x->lo, len_y = y->hi - y->lo;
        Poly l;
        if (len_x >= len_y && x->left) {
            l = cross(x->left.get(), y, kind);
            Poly r = cross(x->right.get(), y, kind);
            return lift_pair(l, x->right.get(), r, x->left.get());
        }
        if (!y->left) throw std::logic_error("partial singleton rectangle");
        l = cross(x, y->left.get(), kind);
        Poly r = cross(x, y->right.get(), kind);
        return lift_pair(l, y->right.get(), r, y->left.get());
    }

    // Two rank-one forms with identical denominator domains share the whole
    // staircase recursion.  Only complete rectangles need both products;
    // partial rectangles are lifted to their common denominator once.
    Poly cross_pair(const Node *x, const Node *y, Kind kind,
                    M first_factor, M second_factor) const {
        int state = classify(kind, x, y);
        if (state == 1) {
            return mul_pair_sum(x->P, y->P, first_factor,
                                x->P2, y->P2, second_factor);
        }
        if (state == 0) return {};
        int len_x = x->hi - x->lo, len_y = y->hi - y->lo;
        Poly l;
        if (len_x >= len_y && x->left) {
            l = cross_pair(x->left.get(), y, kind, first_factor, second_factor);
            Poly r = cross_pair(x->right.get(), y, kind, first_factor, second_factor);
            return lift_pair(l, x->right.get(), r, x->left.get());
        }
        if (!y->left) throw std::logic_error("partial singleton rectangle");
        l = cross_pair(x, y->left.get(), kind, first_factor, second_factor);
        Poly r = cross_pair(x, y->right.get(), kind, first_factor, second_factor);
        return lift_pair(l, y->right.get(), r, y->left.get());
    }

    struct FusionNode {
        const Node *x;
        const Node *y;
        int left = -1;
        int right = -1;
        bool split_x = false;
    };

    int build_fusion_plan(vector<FusionNode> &plan, const Node *x,
                          const Node *y, Kind kind, int depth) const {
        const int id = static_cast<int>(plan.size());
        plan.push_back(FusionNode{x, y});
        if (depth == 0 || classify(kind, x, y) != -1) return id;

        const int len_x = x->hi - x->lo;
        const int len_y = y->hi - y->lo;
        if (len_x >= len_y && x->left) {
            const int left = build_fusion_plan(
                plan, x->left.get(), y, kind, depth - 1);
            const int right = build_fusion_plan(
                plan, x->right.get(), y, kind, depth - 1);
            plan[id].left = left;
            plan[id].right = right;
            plan[id].split_x = true;
            return id;
        }
        if (!y->left) throw std::logic_error("partial singleton rectangle");
        const int left = build_fusion_plan(
            plan, x, y->left.get(), kind, depth - 1);
        const int right = build_fusion_plan(
            plan, x, y->right.get(), kind, depth - 1);
        plan[id].left = left;
        plan[id].right = right;
        return id;
    }

    Poly lift_fusion_plan(const vector<FusionNode> &plan,
                          vector<Poly> &value, int id) const {
        const FusionNode &node = plan[id];
        if (node.left < 0) return std::move(value[id]);
        Poly left = lift_fusion_plan(plan, value, node.left);
        Poly right = lift_fusion_plan(plan, value, node.right);
        if (node.split_x)
            return lift_pair(left, node.x->right.get(),
                             right, node.x->left.get());
        return lift_pair(left, node.y->right.get(),
                         right, node.y->left.get());
    }

    vector<M> inverse_series(const vector<M> &f, int need) const {
        if (f.empty() || f[0] == M(0)) throw std::logic_error("noninvertible series");
        vector<M> g(1, f[0].inv());
        int m = 1;
        while (m < need) {
            const int target = static_cast<int>(std::min<long long>(2LL * m, need));
            vector<M> f_prefix(f.begin(),
                f.begin() + std::min<int>(static_cast<int>(f.size()), target));
            vector<M> correction = convolve_prefix<MOD, ROOT>(
                std::move(f_prefix), g, target);
            correction.resize(target, M(0));
            for (M &coefficient : correction) coefficient = -coefficient;
            correction[0] += M(2);
            g = convolve_prefix<MOD, ROOT>(g, std::move(correction), target);
            g.resize(target);
            m = target;
        }
        return g;
    }

    void add_rational_pair(vector<M> &first_dst, Poly first_numerator,
                           vector<M> &second_dst, Poly second_numerator,
                           Poly denominator) const {
        if (denominator.shift != 0 || denominator.a.empty() || denominator.a[0] != M(1))
            throw std::logic_error("bad root denominator");
        vector<M> inv = inverse_series(denominator.a, N + 1);
        const Poly inverse{0, std::move(inv)};
        auto [first_quotient, second_quotient] =
            mul_two_shared_right(first_numerator, second_numerator, inverse);
        const auto accumulate = [&](vector<M> &dst, const Poly &quotient) {
            for (int i = 0; i < static_cast<int>(quotient.a.size()); ++i) {
                const int degree = quotient.shift + i;
                if (degree <= N) dst[degree] += quotient.a[i];
            }
        };
        accumulate(first_dst, first_quotient);
        accumulate(second_dst, second_quotient);
    }

    Spec spec(int lo, int hi, int den, int slope, long long scale = 1,
              bool by_index = false, int active_lo = -1) const {
        if (active_lo < 0) active_lo = lo;
        return Spec{lo, hi, active_lo, den, slope, scale, by_index, 0, 0, -1};
    }

    std::pair<vector<M>, vector<M>> build_H0_C0() const {
        vector<M> zero(N + 1);
        if (K < 2) return {zero, zero};

        Spec Ai = spec(1, K, 1, 1, 1, false, 2);
        Spec Aj = spec(1, 2 * K, 1, 1);
        Spec Bi = spec(1, K, 2, 1, 1, false, 2);
        Spec iBi = spec(1, K, 2, 1, 1, true, 2);

        vector<M> H0(N + 1), C0(N + 1);

        // W=A-A^(K); combine its A component with the small wedge term.
        Spec Ak = spec(1, 2 * K, 1, K);
        Spec negAk = spec(1, 2 * K, 1, K, -1);
        Spec jAk = spec(1, 2 * K, 1, K, 1, true);
        Spec Q1 = spec(1, K, 1, K + 1, 1, false, 2);
        Spec Q2 = spec(1, K, 2, K + 1, 1, false, 2);
        Spec iQ1 = spec(1, K, 1, K + 1, 1, true, 2);
        Spec iQ2 = spec(1, K, 2, K + 1, 1, true, 2);

        // The triangle y-domain [1,K] is the left half of the wedge tree.
        auto y_full = build_denominator(1, 2 * K, 1);
        auto x = alias_denominator_tree(y_full->left.get());
        vector<FusionNode> wedge_plan;
        wedge_plan.reserve(size_t(1) << (STAIRCASE_FUSION_DEPTH + 1));
        build_fusion_plan(wedge_plan, x.get(), y_full.get(), Kind::Wedge,
                          STAIRCASE_FUSION_DEPTH);
        vector<FusionNode> triangle_plan;
        triangle_plan.reserve(size_t(1) << (STAIRCASE_FUSION_DEPTH + 1));
        build_fusion_plan(triangle_plan, x.get(), y_full->left.get(),
                          Kind::Triangle, STAIRCASE_FUSION_DEPTH);
        Poly h_numerator;
        {
            // Fuse the two H-wedge terms at a common shallow frontier.
            fill_numerator(x.get(), Ai);
            fill_numerator(y_full.get(), Ak);
            vector<Poly> frontier(wedge_plan.size());
            for (size_t k = 0; k < wedge_plan.size(); ++k)
                if (wedge_plan[k].left < 0)
                    frontier[k] = cross(wedge_plan[k].x, wedge_plan[k].y,
                                        Kind::Wedge);
            fill_numerator(y_full.get(), Aj);
            for (size_t k = 0; k < wedge_plan.size(); ++k)
                if (wedge_plan[k].left < 0)
                    frontier[k] = add(frontier[k],
                        cross(wedge_plan[k].x, wedge_plan[k].y, Kind::Wedge));
            Poly numerator = lift_fusion_plan(wedge_plan, frontier, 0);
            // Fuse the two H-triangle terms on the same frontier.
            fill_numerator(x.get(), Q1);
            vector<Poly> triangle_frontier(triangle_plan.size());
            for (size_t k = 0; k < triangle_plan.size(); ++k)
                if (triangle_plan[k].left < 0)
                    triangle_frontier[k] = cross(
                        triangle_plan[k].x, triangle_plan[k].y,
                        Kind::Triangle);
            fill_numerator(y_full->left.get(), negAk);
            for (size_t k = 0; k < triangle_plan.size(); ++k)
                if (triangle_plan[k].left < 0)
                    triangle_frontier[k] = add(triangle_frontier[k], cross(
                        triangle_plan[k].x, triangle_plan[k].y,
                        Kind::Triangle));
            Poly triangle = lift_fusion_plan(
                triangle_plan, triangle_frontier, 0);
            numerator = add(numerator, mul(triangle, y_full->right->D));
            h_numerator = divide_by_one_minus_x(std::move(numerator));
        }

        // Reuse the squared y tree for the identical x interval structure.
        auto y_squared = build_squared_denominator(y_full->left.get());
        retarget_denominator_alias(x.get(), y_squared.get());
        {
            // Reuse -A^(K) in the left subtree and build only the right half.
            fill_numerator(y_full->right.get(), negAk);
            y_full->P = lift_pair(y_full->left->P, y_full->right.get(),
                                  y_full->right->P, y_full->left.get());
            fill_second_numerator(y_full.get(), jAk);
            fill_numerator(x.get(), iBi);
            fill_second_numerator(x.get(), Bi);

            // Merge the two C-wedge shift families at a shallow frontier.
            vector<Poly> frontier(wedge_plan.size());
            for (size_t k = 0; k < wedge_plan.size(); ++k)
                if (wedge_plan[k].left < 0)
                    frontier[k] = cross_pair(wedge_plan[k].x, wedge_plan[k].y,
                                             Kind::Wedge, M(-2), M(-1));
            fill_numerator(y_full.get(), Aj);
            fill_indexed_A_from_denominator(y_full.get());
            for (size_t k = 0; k < wedge_plan.size(); ++k)
                if (wedge_plan[k].left < 0)
                    frontier[k] = add(frontier[k],
                        cross_pair(wedge_plan[k].x, wedge_plan[k].y,
                                   Kind::Wedge, M(2), M(-1)));
            Poly numerator = lift_fusion_plan(wedge_plan, frontier, 0);

            // For T(U,V)=sum_{i>j}(i-j)U_i V_j we have
            //   T(U,V)=T(V,U)+(sum iU_i)(sum V_j)-(sum U_i)(sum jV_j).
            // This transposes the R terms and leaves one closed correction.
            Spec R12 = spec(1, K, 2, 1, K - 1LL);
            R12.second_slope = 2;
            R12.second_scale = -K;
            Spec iR12 = R12;
            iR12.multiply_by_index = true;

            // Fuse the two recursive triangle pairs before lifting the root.
            fill_numerator(x.get(), iQ2);
            fill_second_numerator(x.get(), Q2);
            vector<Poly> triangle_frontier(triangle_plan.size());
            for (size_t k = 0; k < triangle_plan.size(); ++k)
                if (triangle_plan[k].left < 0)
                    triangle_frontier[k] = cross_pair(
                        triangle_plan[k].x, triangle_plan[k].y,
                        Kind::Triangle, M(1), M(-1));

            // Root numerators for the final monomial of R.
            Poly iR_last = x->P;
            Poly R_last = x->P2;

            // For j>=2 the adjacent pair cancels its denominator exactly:
            //   (-q^(Kj)+q^((K+1)j))/(1-q^j) = -q^(Kj).
            // The remaining triangle is a finite polynomial whose numerator
            // is divisible by (1-q^K)^2; j=1 is added explicitly below.
            Spec shifted_Q2 = spec(1, K, 2, 2 * K + 1, 1, false, 2);
            fill_numerator(x.get(), shifted_Q2);
            const Poly shifted_root = x->P;
            Poly finite_numerator = sub(shift_by(iR_last, K),
                                        shift_by(R_last, K));
            finite_numerator = sub(std::move(finite_numerator),
                                   shift_by(iR_last, 2 * K));
            finite_numerator = add(finite_numerator,
                                   shift_by(shifted_root, K));
            finite_numerator = divide_by_one_minus_q_power(
                divide_by_one_minus_q_power(std::move(finite_numerator), K), K);
            finite_numerator = scalar(std::move(finite_numerator), M(-1));
            Poly triangle_extra = mul(finite_numerator, y_full->left->D);

            Poly exceptional = shift_by(sub(iR_last, R_last), K + 1);
            exceptional = scalar(std::move(exceptional), M(-1));
            Poly without_f1 = divide_by_one_minus_x(y_full->left->D);
            triangle_extra = add(triangle_extra,
                                 mul(exceptional, without_f1));

            fill_numerator(x.get(), iR12);
            fill_numerator(y_full->left.get(), Q1);
            fill_second_numerator(x.get(), R12);
            fill_second_numerator(y_full->left.get(), iQ1);
            for (size_t k = 0; k < triangle_plan.size(); ++k)
                if (triangle_plan[k].left < 0)
                    triangle_frontier[k] = add(triangle_frontier[k],
                        cross_pair(triangle_plan[k].x, triangle_plan[k].y,
                                   Kind::Triangle, M(1), M(-1)));
            Poly triangle = add(
                lift_fusion_plan(triangle_plan, triangle_frontier, 0),
                triangle_extra);

            // Add the index-1 leaf omitted from Q2.
            const Spec R_last_full = spec(1, K, 2, K + 1);
            const Poly R_at_one = lift_single_leaf(x.get(), 1, R_last_full);
            const Poly R_root = add(add(x->P2, R_last), R_at_one);
            const Poly iR_root = add(add(x->P, iR_last), R_at_one);
            const Poly correction = mul_pair_sum(
                y_full->left->P2, R_root, M(1),
                y_full->left->P, iR_root, M(-1));
            triangle = add(triangle, correction);
            numerator = add(numerator, mul(triangle, y_full->right->D));
            Poly denominator = mul(denominator_poly(x.get()),
                                   denominator_poly(y_full.get()));
            numerator = divide_by_one_minus_x(std::move(numerator));
            denominator = divide_by_one_minus_x(std::move(denominator));
            // Lift H0 to the common denominator and share one Newton inverse.
            h_numerator = mul(h_numerator, y_full->left->D);
            add_rational_pair(H0, std::move(h_numerator),
                              C0, std::move(numerator), std::move(denominator));
        }
        return {std::move(H0), std::move(C0)};
    }

  public:
    Solver(int n, const vector<int> &prime_list)
        : N(n), K(static_cast<int>(std::sqrt(static_cast<long double>(n)))),
          primes(prime_list) {
        while (1LL * (K + 1) * (K + 1) <= N) ++K;
        while (1LL * K * K > N) --K;
    }

    std::pair<vector<M>, vector<M>> run() const {
        auto [H0, C0] = build_H0_C0();
        vector<int> mu(N + 1, 1);
        mu[0] = 0;
        for (int p : primes) {
            for (int multiple = p; multiple <= N; multiple += p)
                mu[multiple] = -mu[multiple];
            if (p <= N / p)
                for (int multiple = p * p; multiple <= N; multiple += p * p)
                    mu[multiple] = 0;
        }

        vector<M> primitive_H(N + 1), primitive_C(N + 1);
        for (int divisor = 1; divisor <= N; ++divisor) {
            if (mu[divisor] == 0) continue;
            const M h_factor = M(mu[divisor]);
            const M c_factor = h_factor * M(divisor);
            // Bound source before forming the product.  This also avoids
            // signed overflow in destination += divisor at N=2^30 when the
            // outer loop reaches divisor=N.
            for (int source = 1; source <= N / divisor; ++source) {
                const int destination = source * divisor;
                primitive_H[destination] += h_factor * H0[source];
                primitive_C[destination] += c_factor * C0[source];
            }
        }
        return {std::move(primitive_H), std::move(primitive_C)};
    }
};

static vector<int> primes_up_to(int limit) {
    vector<bool> composite(limit + 1);
    vector<int> primes;
    for (int i = 2; i <= limit; ++i) {
        if (composite[i]) continue;
        primes.push_back(i);
        if (i <= limit / i)
            for (int multiple = i * i; multiple <= limit; multiple += i)
                composite[multiple] = true;
    }
    return primes;
}

template <uint64_t MOD, uint64_t ROOT>
static vector<uint64_t> modular_solution(
        int n, const vector<int> &primes) {
    Solver<MOD, ROOT> solver(n, primes);
    auto [h, c] = solver.run();
    using M = Mint<MOD>;
    const M two(2), four(4), inv_six = M(6).inv();
    M sum_x, sum_mx, sum_m2x, sum_y, sum_my;
    vector<uint64_t> residue(n + 1);
    for (int index = 1; index <= n; ++index) {
        const M m(index);
        const M axis = m * M(index - 1) * M(index - 1)
                     * M(2LL * index - 1) * inv_six;
        const M value = axis + m * m * sum_x - two * m * sum_mx
                       + sum_m2x + m * sum_y - sum_my;
        residue[index] = value.value();

        const M x = four * h[index] + two * M((index - 1) / 2);
        const M y = four * c[index];
        sum_x += x;
        sum_mx += m * x;
        sum_m2x += m * m * x;
        sum_y += y;
        sum_my += m * y;
    }
    return residue;
}

static uint64_t mod_power(uint64_t a, uint64_t exponent, uint64_t modulus) {
    uint64_t result = 1;
    while (exponent != 0) {
        if (exponent & 1)
            result = static_cast<uint64_t>((__uint128_t(result) * a) % modulus);
        a = static_cast<uint64_t>((__uint128_t(a) * a) % modulus);
        exponent >>= 1;
    }
    return result;
}

static void crt_extend(vector<BigUInt> &answer, BigUInt &current_modulus,
                       const vector<uint64_t> &residue, uint64_t modulus) {
    if (answer.empty()) {
        answer.reserve(residue.size());
        for (uint64_t value : residue) answer.emplace_back(value);
        current_modulus = BigUInt(modulus);
        return;
    }
    const uint64_t base = current_modulus.mod_small(modulus);
    const uint64_t inverse = mod_power(base, modulus - 2, modulus);
    for (size_t i = 1; i < answer.size(); ++i) {
        const uint64_t old = answer[i].mod_small(modulus);
        const uint64_t difference = residue[i] >= old
            ? residue[i] - old : modulus - (old - residue[i]);
        const uint64_t digit = static_cast<uint64_t>(
            (__uint128_t(difference) * inverse) % modulus);
        answer[i].add_multiple(current_modulus, digit);
    }
    current_modulus.multiply_small(modulus);
}

static vector<BigUInt> compute_all_F(int N) {
    vector<int> primes = primes_up_to(N);
    vector<BigUInt> exact;
    BigUInt crt_modulus(1);
    crt_extend(exact, crt_modulus,
               modular_solution<NTT_MOD_1, NTT_ROOT>(N, primes), NTT_MOD_1);
    crt_extend(exact, crt_modulus,
               modular_solution<NTT_MOD_2, NTT_ROOT>(N, primes), NTT_MOD_2);

    // Exact published value F(2^30), used only to certify the fixed CRT range.
    BigUInt maximum;
    maximum.limb = {14705131133890331560ULL, 263019320687255696ULL};
    if (crt_modulus <= maximum)
        throw std::runtime_error("CRT modulus product is too small");
    return exact;
}

static BigUInt brute_F(int n) {
    __uint128_t ans = __uint128_t(n) * (n - 1) * (n - 1) * (2 * n - 1) / 6;
    for (int u = 2; u <= n; ++u) {
        for (int v = 1; v < u; ++v) {
            if (std::gcd(u, v) != 1) continue;
            for (int x = 1; x * u <= n; ++x) {
                for (int y = 1; y <= x; ++y) {
                    const int L = x * u + y * v;
                    if (L > n) break;
                    const int K = x * v + y * u;
                    const int multiplicity = (x == y ? 2 : 4);
                    ans += __uint128_t(multiplicity) * (n - L) * (n - K);
                }
            }
        }
    }
    BigUInt result;
    result.limb = {static_cast<uint64_t>(ans), static_cast<uint64_t>(ans >> 64)};
    return result;
}

static bool self_test() {
    const vector<BigUInt> fast = compute_all_F(96);
    for (int n = 1; n <= 96; ++n) {
        const BigUInt slow = brute_F(n);
        if (fast[static_cast<size_t>(n)].limb != slow.limb) {
            std::cerr << "self-test failed at N=" << n
                      << " fast=" << fast[static_cast<size_t>(n)].str()
                      << " brute=" << slow.str() << '\n';
            return false;
        }
    }
    std::cerr << "self-test OK for N=1..96\n";
    return true;
}

static bool parse_int(const char *text, int &value) {
    try {
        size_t parsed = 0;
        const long long result = std::stoll(text, &parsed);
        if (parsed != std::string(text).size()
            || result < std::numeric_limits<int>::min()
            || result > std::numeric_limits<int>::max())
            return false;
        value = static_cast<int>(result);
        return true;
    } catch (const std::exception &) {
        return false;
    }
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

int main(int argc, char **argv) {
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
            const vector<BigUInt> answer = compute_all_F(N);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << power << '\t' << N << '\t' << trial << '\t'
                      << seconds << '\t' << answer[(size_t)N].str() << '\n';
            std::cout.flush();
        }
    }
    return 0;
}
