### Declare custom types
## Abstract types
abstract type AbstractStatisticalManifold{ℝ} <: AbstractDecoratorManifold{ℝ} end
abstract type AbstractStatisticalModel end#<: Distribution end

function active_traits(f,::AbstractStatisticalManifold,args...)
    return merge_traits(IsEmbeddedManifold(),IsDefaultMetric(FisherRaoMetric()))
end
## Create concrete instances of custom types]
struct StatisticalModel <: AbstractStatisticalModel
    callable_distribution::Function
    natural_parameter_manifold::AbstractManifold
    parameterdimension::Int
    function StatisticalModel(D, M, pd)
        return !(typeof(D()) <: Distribution) ?
               error(
            "Model function evaluated at the null argument should be an object of type <:Distribution",
        ) : new(D, M, pd)
    end
end
### Outer constructor methods
StatisticalModel(D::Function,M::AbstractManifold) = StatisticalModel(D,M,manifold_dimension(M))

@kwdef struct StatisticalManifold <: AbstractStatisticalManifold{ℝ}
    model::StatisticalModel
    hypermodel::Distribution
    parametermap
    embedded::Bool = false
    function StatisticalManifold(x, y, z, E)
        return !(typeof(x) <: StatisticalModel) || !(typeof(y) <: Distribution) ?
               error("Arguments should be objects of type <:Distribution") : new(x, y, z, E)
    end
end
#Outer constructor methods


struct Probability{T<:AbstractFloat} <: AbstractFloat 
    x::T
    function Probability(x::Number)
        if (x<0) || (x>1)
            throw(ArgumentError("x is $x but must be in the range [0, 1]"))
        end
        return new{typeof(x)}(x)
    end
end


### New Alias for structures in dependencies
const CountablyDiscreteDistribution = Union{
    Bernoulli,BernoulliLogit,BetaBinomial,Binomial,Categorical,Dirac,DiscreteUniform,
    Hypergeometric,PoissonBinomial,Soliton,Multinomial
}
const UncountablyDiscreteDistribution = Union{
    Geometric,NegativeBinomial,Poisson,Skellam
}
const PDFDistribution = Union{
    UnivariateDistribution,MatrixDistribution,UnivariateMixture,MultivariateMixture
}
const LogLikelihoodDistribution = MultivariateDistribution
### Conversion functions
function (::Type{T})(x::Probability) where {T<:AbstractFloat}
    return convert(T,x.x)
end
function Base.promote_rule(::Type{T}, ::Type{Probability{S}}) where {T<:Number,S}
    return promote_type(T,S)
end