function [b] = is_int_val(x)
b = isfinite(x) & x == floor(x);
