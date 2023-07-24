module AlgorithmX

export exact_cover

"""
    exact_cover(X::Matrix{Bool})

Solve the exact cover problem using Knuth's Algorithm X. The input is a Boolean
matrix `X` whose columns encode the universe and whose rows encode the subsets.

The algorithm halts when it has found a single solution. If no solution is
found, an empty set is returned.

# Example
Consider a universe `{1, 2, 3, 4, 5, 6, 7}` and a collection of sets `{A, B, C,
D, E, F}` where
* `A = {1, 4, 7}`;
* `B = {1, 4}`;
* `C = {4, 5, 7}`;
* `D = {3, 5, 6}`;
* `E = {2, 3, 6, 7}`
* `F = {2, 7}`.
It has a unique exact subcover, namely `{B, D, F}`.

To enter the problem into `exact_cover`, convert it into a matrix:
```julia-repl  
julia> X = Matrix{Bool}([
    1 0 0 1 0 0 1;
    1 0 0 1 0 0 0;
    0 0 0 1 1 0 1;
    0 0 1 0 1 1 0;
    0 1 1 0 0 1 1;
    0 1 0 0 0 0 1
]);
```
Now evaluate:
```
julia> exact_cover(M)
3-element Vector{Int64}:
 2
 4
 6
```
Note that these are the indices of the subsets `B`, `D`, and `F`, as desired.
"""
function exact_cover(X::Matrix{Bool})
    solution = Vector{Int64}([])
    function solve(X::AbstractMatrix{Bool})
        # If a matrix has no columns, that means that all columns have been
        # covered, and the algorithm terminates successfully.
        size(X, 2) == 0 && return true
        # The algorithm essentially brute-forces through the rows and columns.
        # In principle, we could proceed through them one by one in the order
        # presented, but in practice, it helps to reorder the columns by the
        # number of 1's first.
        col_order = sort(1:size(X, 2), by = i -> sum(X[:, i]))
        # If one of the columns has only zeros, it cannot be covered, so the
        # algorithm terminates unsuccessfully. Since we've sorted the columns
        # by number of 1's, it suffices to consider the first column.
        sum(X[:, col_order[1]]) == 0 && return false
        for c in col_order
            # Arbirarily try out a column.
            selected_col = @view X[:, c]
            for r in findall(selected_col)
                # Arbirarily try out a row.
                selected_row = @view X[r, :]
                # The row chosen covers a specific set of columns; these columns
                # need no longer be considered in the next iteration of solve().
                subcols = .!selected_row
                # Any row which has a 1 in any of the columns that we removed in
                # the line above, must be removed, because otherwise a column
                # ends up being covered by more rows.
                subrows = [!any(X[r, selected_row]) for r in axes(X, 1)]
                push!(solution, parentindices(X)[1][r])
                solve(@view X[subrows, subcols]) && return true
                # If the recursive solve() in the above line returned without
                # success, that means our guessed row and column were wrong, so
                # we remove the latest element in our solution list of indices.
                pop!(solution)
            end
        end
    end
    solve(X)
    return solution
end

end