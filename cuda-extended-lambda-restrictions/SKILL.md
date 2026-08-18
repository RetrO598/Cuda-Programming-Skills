---
name: cuda-extended-lambda-restrictions
description: >
  Check CUDA C++ and Kokkos code for NVIDIA CUDA extended-lambda restrictions. Use when code contains or may generate extended __device__ or __host__ __device__ lambdas, including CUDA builds involving KOKKOS_LAMBDA. Focus only on the CUDA Programming Guide Extended Lambda Restrictions and related compile failures.
---

# CUDA Extended Lambda Restrictions

## Scope

Use this skill only to detect and fix CUDA **extended lambda** issues described in:

NVIDIA CUDA Programming Guide **5.3.8.4 Extended Lambda Restrictions**   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#extended-lambda-restrictions

An extended lambda is a CUDA lambda with explicit `__device__` or `__host__ __device__` annotation that satisfies CUDA's extended-lambda definition. When Kokkos code is compiled in a mode where a `KOKKOS_LAMBDA` becomes such a CUDA extended lambda, apply the same restrictions.

NVCC may replace an extended lambda with a namespace-scope placeholder type in code sent to the host compiler. Many restrictions below follow from that transformation.

---

## Review procedure

When relevant code is encountered:

1. Identify each extended device-only or host-device lambda.
2. Identify its CUDA **enclosing function**.
3. Check Rules 1–18 below.
4. If a rule is violated, make the smallest change that removes the violation.
5. Do not invent additional CUDA restrictions not listed here.
6. Distinguish an explicit CUDA restriction from an ordinary C++ or Kokkos issue.

---

# Rules

## 1. No extended lambda inside another extended lambda

Reject:

```cpp
auto outer = [] __host__ __device__ {
    auto inner = [] __device__ {};
};
```

Move the inner callable outside the extended lambda or use a named callable.

## 2. No extended lambda inside a generic lambda

Reject an extended lambda defined inside a lambda whose parameter list is generic, for example one using `auto`.

```cpp
auto outer = [](auto x) {
    auto inner = [] __device__ {};
};
```

## 3. Nested ordinary lambdas must ultimately be inside a function

An extended lambda may be nested inside ordinary lambdas only when the outermost enclosing lambda itself is defined within the immediate or nested block scope of a function.

Do not place such a nested structure under a namespace/global-scope lambda.

## 4. The enclosing function must be named and addressable

The CUDA compiler must be able to take the address of the enclosing function.

For a member function:

- every enclosing class must have a name;
- the member function must not be `private` or `protected`;
- enclosing classes must not be inaccessible through `private` or `protected`
  nesting.

Constructors and inaccessible member functions can therefore make an extended lambda invalid when CUDA cannot form the required enclosing-function address.

## 5. The enclosing-function address must be unambiguous

At the extended-lambda definition point, NVCC must be able to form the address of the enclosing routine unambiguously.

Be cautious with template aliases, shadowing, or declarations that can cause the generated address expression to denote a different specialization.

## 6. No extended lambda in a function-local class

Reject an extended lambda defined in a member function of a class declared locally inside another function.

```cpp
void f() {
    struct Local {
        void run() {
            auto x = [] __device__ {};
        }
    };
}
```

Move the class to namespace or normal class scope.

## 7. The enclosing function cannot have a deduced return type

Reject:

```cpp
auto f() {
    auto x = [] __device__ {};
}
```

when this function is the CUDA enclosing function.

Give the enclosing function an explicit return type.

## 8. A host-device extended lambda cannot be generic

Reject:

```cpp
auto f = [] __host__ __device__ (auto x) {
    return x;
};
```

This restriction applies to extended `__host__ __device__` lambdas.

Do not incorrectly generalize this rule to all device-only extended lambdas.

## 9. Template enclosing functions have additional constraints

If the enclosing function is:

- a function-template instantiation;
- a member-function-template instantiation; or
- a member of a class template,

check all of the following:

- at most one variadic template parameter is present;
- if present, the variadic template parameter is last;
- all template parameters are named;
- template-instantiation argument types are not function-local types, except
  permitted extended-lambda closure types;
- template-instantiation argument types are not inaccessible `private` or
  `protected` member types.

Do not simplify this to a blanket ban on templates.

## 10. Captured-variable restrictions

Apply all of the following.

### 10.1 Capture only by value

Reject reference capture:

```cpp
auto f = [&x] __device__ { return x; };
```

and reference init-capture.

### 10.2 Captured arrays are limited to seven dimensions

A captured C-style array cannot have more than seven dimensions.

### 10.3 Captured array elements need host default construction and copy assignment

For captured array variables, CUDA's host-side transformation can default-initialize the closure array field and then copy-assign its elements.

Therefore the array element type must be:

- default-constructible in host code;
- copy-assignable in host code.

### 10.4 Do not capture a function parameter that is an element of a variadic argument pack

Refactor the required value before creating the extended lambda.

### 10.5 Captured variable types cannot be prohibited local/inaccessible types

A captured variable's type cannot be local to a function, except for permitted extended-lambda closure types, and cannot be an inaccessible `private` or `protected` class member type.

### 10.6 Init-capture differs for device-only and host-device lambdas

For an extended host-device lambda, init-capture is not supported.

Reject:

```cpp
auto f = [x = expr] __host__ __device__ {};
```

For an extended device-only lambda, init-capture is supported except when the initializer is:

- an array; or
- `std::initializer_list`.

### 10.7 Extended lambdas are not `constexpr`/`consteval` lambdas

The extended lambda call operator is not `constexpr`, its closure is not a literal type, and the extended lambda declaration cannot use `constexpr` or `consteval`.

### 10.8 No first implicit capture only inside nested `if constexpr`

A variable cannot become implicitly captured for the first time only inside an `if constexpr` lexically nested in the extended lambda.

Unsafe:

```cpp
int var = 4;

auto f = [=] __device__ {
    if constexpr (Condition) {
        use(var);
    }
};
```

Fix by explicitly capturing `var`, or ensure it is already implicitly captured outside the nested `if constexpr`.

## 11. Extended-lambda count/order must not depend on `__CUDA_ARCH__`

NVCC assigns a counter to extended lambdas when parsing a function.

Therefore the presence, absence, number, or relative declaration order of extended lambdas in a function must not depend on:

- whether `__CUDA_ARCH__` is defined;
- the value of `__CUDA_ARCH__`.

Reject preprocessor branches that alter this structure.

## 12. Host-side `operator()` introspection may be invalid for device-only extended lambdas

For a device-only extended lambda defined in host code, the host placeholder normally does not expose an `operator()` equivalent to the original lambda.

Do not assume host code can determine the lambda call operator's return type or parameter types.

The exception is a device-only extended lambda for which:

```cpp
__nv_is_extended_device_lambda_with_preserved_return_type(...)
```

is true.

Host-device extended lambdas are not subject to this device-only restriction.

## 13. Device-only `operator()` introspection rules

For an extended device-only lambda:

- parameter-type introspection of `operator()` is supported only in device code;
- return-type introspection is supported only in device code unless
  `__nv_is_extended_device_lambda_with_preserved_return_type(...)` is true.

When generic host code attempts `result_of`, `invoke_result`, or equivalent callable-signature inspection, check this rule first.

## 14. Capturing expressions must not change with `__CUDA_ARCH__`

If an extended lambda is passed from host to device code, expressions in its body that cause captures must remain structurally unchanged across host/device compilation.

Reject code where `__CUDA_ARCH__` changes which variables are captured or their encounter order.

Otherwise host and device closure layouts can differ.

## 15. Device-only extended lambda cannot convert to a function pointer in host code

A suitable extended device-only lambda can have a lambda-to-function-pointer conversion in device code.

Do not perform that conversion in host code.

```cpp
auto d = [] __device__ (double) { return 1; };

// invalid in host code
int (*fp)(double) = d;
```

This restriction does not apply in the same way to an extended host-device lambda, whose host-side conversion is supported.

## 16. Do not use affected triviality traits to select CUDA template instantiations

The CUDA front end and host compiler may disagree on these traits for an extended-lambda closure:

```cpp
std::is_trivially_copyable
std::is_trivially_constructible
std::is_trivially_copy_constructible
std::is_trivially_move_constructible
std::is_trivially_destructible
```

Do not use results of these traits on an extended-lambda closure to instantiate `__global__`, `__device__`, `__constant__`, or `__managed__` function or variable templates.

Prefer an explicit policy/template argument or a stable named application type.

---

# Diagnostics

For later restrictions, especially placeholder-type and host/device disagreement cases, the failure may instead appear as a host-compiler error or incorrect generated behavior.

When diagnostics mention unfamiliar NVCC-generated wrapper or placeholder types, check Rules 13–16 before rewriting unrelated generic code.

# Output behavior

When this skill detects a relevant issue:

1. cite the violated rule number;
2. explain the CUDA-specific reason briefly;
3. provide the smallest legal rewrite;
4. do not add unrelated CUDA/Kokkos optimization advice unless explicitly asked.
