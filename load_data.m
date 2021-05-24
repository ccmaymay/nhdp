function [X] = load_data(path)
X = spconvert(dlmread(path));
