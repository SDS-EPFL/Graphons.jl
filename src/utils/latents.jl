## Mapping continuous latents to discrete block labels and vice versa

function _convert_latent_to_block(sbm, ξ)
    return searchsortedfirst(sbm.cumsize, ξ)
end


@inline function _convert_latent_to_block(sbm, ξ1, ξ2)
    return _convert_latent_to_block(sbm, ξ1), _convert_latent_to_block(sbm, ξ2)
end


function node_labels_to_latents(
    node_labels::AbstractVector{Int},
    sbm::Union{SBM,DecoratedSBM,SSM},
)
    return map(label -> _label_to_latent(label, sbm), node_labels)
end

function _label_to_latent(label::Int, sbm::Union{SBM,DecoratedSBM,SSM})
    return sbm.cumsize[label] - eps()
end


function permute!(sbm::Union{SBM,DecoratedSBM,SSM}, perm)
    sbm.size .= sbm.size[perm]
    sbm.cumsize .= cumsum(sbm.size)
    sbm.θ .= sbm.θ[perm, perm]
    return nothing
end
