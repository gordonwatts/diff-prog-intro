# Exploring Differentiable Programming

I have read a bunch about differentiable programming. So I sort-of know what I'm talking about in broad strokes.
However, I don't really know what it takes. How much data do you have to move around the system? What are the actual
operations. How do you make a selection cut that is differentiable. How do you use JAX?

This book is me trying to teach myself step-by-step. So it is very basic! Comments and pull requests are welcome at the `github` repo!

```{tableofcontents}
```

## Goals

* Construct a simple $S/\sqrt{B}$ problem that needs to be optimiszed - a single selection cut, and a single signal and background. Solve it by brute force.
* Learn the basics of `JAX` as a `numpy` replacement.
* Figure out how to make a hard selection cut (`data[data > cut]`) differentiable w.r.t `cut`.
* Write a very simple gradient decent loop using `JAX` tools to solve this problem.

