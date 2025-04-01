### Numerical implementations
function divergence(d1::T,d2::T,divtype::Function="symmKL";kwargs...) where {T<:Distribution}
    divtype(d1,d2;kwargs...)
end
function KL(d1::T,d2::T;kwargs...) where {T<:Distribution} 
    P(x) = pdf(d1,big(x));Q(x) = pdf(d2,big(x));
    integrand(x) = Float64(log(10,P(x)/Q(x)));
    div = expectation(x->integrand(x),d1;kwargs...)
    return div
end
function symmKL(d1::T,d2::T;kwargs...) where {T<:Distribution}
    P(x) = pdf(d1,big(x));Q(x) = pdf(d2,big(x)); 
    integrand1(x) = Float64(log(10,P(x)/Q(x)));integrand2(x) = Float64(log(10,Q(x)/P(x)));
    div = expectation(x->integrand1(x),d1;kwargs...) + expectation(x->integrand2(x),d2;kwargs...)
    return div
end
function TsallisKL(d1::T,d2::T;q::AbstractFloat=1,kwargs...) where {T<:Distribution} 
    P(x) = pdf(d1,big(x));Q(x) = pdf(d2,big(x));
    integrand(x) = Float64(TsallisLog(P(x)/Q(x),p=Probability(q)));
    div = expectation(x->integrand(x),d1;kwargs...)
    return div
end
function Renyi(d1::T,d2::T;beta::Real=0.5,kwargs...) where {T<:Distribution} 
    P(x) = pdf(d1,big(x));Q(x) = pdf(d2,big(x));
    integrand(x) = Float64((P(x)*Q(x))^(beta-1));
    if (beta<0)||(beta==1)
        throw(ArgumentError("beta is $(beta) but must be in the range (0, 1)∪(1, ∞)"))
    end
    div = log(expectation(x->integrand(x),d1;kwargs...))/(1-beta)
    return div
end
function JS(d1::T,d2::T;kwargs...) where {T<:Distribution} 
    P(x) = pdf(d1,big(x));Q(x) = pdf(d2,big(x));M(x) = (P(x)+Q(x))/2;
    integrand1(x) = Float64(log(10,P(x)/M(x)));integrand2(x) = Float64(log(10,Q(x)/M(x)));
    div = expectation(x->integrand1(x),d1;kwargs...) + expectation(x->integrand2(x),d2;kwargs...)
    return div
end
function Bregman(d1::T,d2::T;f::Function,difference::Function) where {T<:Distribution}
    theta1 = params(d1);theta2 = params(d2);
    Gradf(x) = ForwardDiff.gradient(f,x)
    div = f(theta1) - f(theta2) - difference(theta1,theta2).*Gradf(theta2)
    return div
end
