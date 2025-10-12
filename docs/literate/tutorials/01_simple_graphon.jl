
h = 300; # hide

# Define a simple step-function graphon
W(u, v) = u * v

# We can visualize this graphon as a heatmap.
let
    grid = 0:0.01:1
    fig = Mke.Figure(size=(h + 20, h))
    ax = Mke.Axis(fig[1, 1], title="True Graphon W(u,v)",
        xlabel="u", ylabel="v", aspect=Mke.DataAspect())
    hm = Mke.heatmap!(ax, grid, grid, W, colormap=:binary, colorrange=(0, 1))
    Mke.Colorbar(fig[1, 2], hm)
    fig
end
