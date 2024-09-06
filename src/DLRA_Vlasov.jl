module DLRA_Vlasov

using LinearAlgebra
using FFTW
using OrdinaryDiffEq
using MutableNamedTuples
using ProgressMeter
using DelimitedFiles
using FStrings
using LaTeXStrings
using LegendrePolynomials
const mnt = MutableNamedTuple

alg() = Vern7()

################## 1D #################

function A_dot(Y0::AbstractArray,p,t)
    """
        A_dot(Y,p,t)

    Right-hand side of the Vlasov-Poisson equation.
    # Arguments
    - `Y0::AbstractArray`: the initial data.
    - `p::NamedTuple`: the parameters.
    - `t::AbstractFloat`: the time of evaluation.
    """

    x = p.x
    v = p.v
    E = p.E
    dx(y) = der_fft(y,x)
    dxY = reduce(hcat,dx.(eachcol(Y0)))
    dy(y) = der_fft(y,v)
    dvY = reduce(hcat,dy.(eachrow(Y0)))
    return -v.*dxY + E .* dvY
end


function A_dot(X0::AbstractArray,
               S0::AbstractArray,
               V0::AbstractArray,p,t)
    """
        A_dot(X,S,V,p,t)

    Right-hand side of the Vlasov-Poisson equation, using a low-rank representation.
    # Arguments
    - `X0::AbstractArray`: the left basis matrix with orthonormal columns.
    - `S0::AbstractArray`: the coefficient matrix of the initial data.
    - `V0::AbstractArray`: the right basis matrix with orthonormal columns.
    - `p::NamedTuple`: the parameters.
    - `t::AbstractFloat`: the time of evaluation.
    """


    x = p.x
    v = p.v
    E = p.E
    dxX = der_fft(X0,x)
    dvV = der_fft(V0,v)

    return -dxX*S0*(v.*V0)' + (E .* X0)*S0*dvV'
end

function K_dot!(K̇,K0,p,t)

    """
        K_dot!(K̇,K0,p,t)

    Right-hand side of the DLRA K-step for Vlasov-Poisson equation.
    # Arguments
    - `K̇::AbstractArray`: the matrix to be mutated.
    - `K0::AbstractArray`: the initial data.
    - `p::NamedTuple`: the parameters.
    - `t::AbstractFloat`: the time of evaluation.
    """

    E = p.E
    c1 = p.c1
    c2 = p.c2
    x = p.x
    tmp = p.tmp

    der_fft!(tmp,K0,p.dx)

    @fastmath mul!(K̇,-tmp,c1')
    @fastmath mul!(tmp,Diagonal(E),K0)
    @fastmath mul!(K̇,tmp,c2',1,1)
end

function L_dot!(L̇,L0,p,t)

    """
        L_dot!(L̇,L0,p,t)

    Right-hand side of the DLRA L-step for Vlasov-Poisson equation.
    # Arguments
    - `L̇::AbstractArray`: the matrix to be mutated.
    - `L0::AbstractArray`: the initial data.
    - `p::NamedTuple`: the parameters.
    - `t::AbstractFloat`: the time of evaluation.
    """

    v = p.v
    d1 = p.d1
    d2 = p.d2
    tmp = p.tmp

    der_fft!(tmp,L0,p.dv)

    @fastmath mul!(L̇,tmp,d1')
    @fastmath mul!(tmp,Diagonal(v),L0)
    @fastmath mul!(L̇,-tmp,d2',1,1)

end

function S_dot!(Ṡ,S0,p,t)

    """
        S_dot!(Ṡ,S0,p,t)

    Right-hand side of the DLRA S-step for Vlasov-Poisson equation.
    # Arguments
    - `Ṡ::AbstractArray`: the matrix to be mutated.
    - `S0::AbstractArray`: the initial data.
    - `p::NamedTuple`: the parameters.
    - `t::AbstractFloat`: the time of evaluation.
    """

    c1 = p.c1
    c2 = p.c2
    d1 = p.d1
    d2 = p.d2

    Ṡ .= muladd(-d2,S0*c1',d1*S0*c2')
end

function calc_c1!(c1,v,V0)
    """
        calc_c1!(c1,v,V0)

    Compute the c^1 coefficient matrix for the Vlasov-Poisson equation.
    # Arguments
    - `c1::AbstractArray`: the matrix to be mutated.
    - `v::AbstractArray`: the velocity vector.
    - `V0::AbstractArray`: the initial data.
    """
    @fastmath mul!(c1,V0', v.*V0)
end

function calc_c2!(c2,dv,V0,dvV)
    """
        calc_c2!(c1,dv,V0,dvV)

    Compute the c^2 coefficient matrix for the Vlasov-Poisson equation.
    # Arguments
    - `c2::AbstractArray`: the matrix to be mutated.
    - `dv::Tuple`: a triplet containing a vector to be mutated, a frequency vector, and a pre-planned FFT object.
    - `V0::AbstractArray`: the initial data.
    - `dvV::AbstractArray`: the matrix to be mutated for the spectral computation of the derivative.
    """
    der_fft!(dvV,V0,dv)
    @fastmath mul!(c2,V0',dvV)
end

function calc_d1!(d1,E,X0)
    """
        calc_d1!(d1,E,X0)

    Compute the d^1 coefficient matrix for the Vlasov-Poisson equation.
    # Arguments
    - `d1::AbstractArray`: the matrix to be mutated.
    - `E::AbstractArray`: the electric field vector.
    - `X0::AbstractArray`: the initial data.
    """
    @fastmath mul!(d1, X0', E .* X0)
end

function calc_d2!(d2,dx,X0,dxX)
    """
        calc_d2!(d2,dx,X0,dxX)

    Compute the d^2 coefficient matrix for the Vlasov-Poisson equation.
    # Arguments
    - `d2::AbstractArray`: the matrix to be mutated.
    - `dx::Tuple`: a triplet containing a vector to be mutated, a frequency vector, and a pre-planned FFT object.
    - `X0::AbstractArray`: the initial data.
    - `dxX::AbstractArray`: the matrix to be mutated for the spectral computation of the derivative.
    """
    der_fft!(dxX,X0,dx)
    @fastmath mul!(d2, X0', dxX)
end

function calc_E!(E,X0,S0,V0,h,dx)
    """
        calc_E(X0,S0,V0,k,h)

    Solve the Poisson equation for the inrotational electric field.
    # Arguments
    - `X0::AbstractArray`: the left basis matrix with orthonormal columns.
    - `S0::AbstractArray`: the coefficient matrix of the initial data.
    - `V0::AbstractArray`: the right basis matrix with orthonormal columns.
    - `h::AbstractFloat`: the step-size of the velocity discretization.
    - `dx::Tuple`: a triplet containing a vector to be mutated, a frequency vector, and a pre-planned FFT object.
    """
    p = dx[3]
    ρ̂ = dx[1]
    ρ̂  .= @views p * (1 .- (X0*S0*sum(V0,dims=1)')[:,1] * h )
    Ê = -im * ρ̂ ./ dx[2]
    Ê[1] = zero(eltype(Ê))
    E .= real.(p\Ê)
end

function initialize(N,r,rmax,τ,flag,integrator)

    T = Float64
    Nx = N#÷2
    Nv = N#*2

    if integrator == "mBUG"
        X0 = Array{T}(undef,Nx,4*rmax)
        V0 = Array{T}(undef,Nv,4*rmax)
        S0 = Array{T}(undef,4*rmax,4*rmax)
    else
        X0 = Array{T}(undef,Nx,2*rmax)
        V0 = Array{T}(undef,Nv,2*rmax)
        S0 = Array{T}(undef,2*rmax,2*rmax)
    end
    K0 = similar(X0)
    L0 = similar(V0)
    tmpx = similar(X0)
    tmpv = similar(V0)
    c1 = similar(S0)
    c2 = similar(S0)
    d1 = similar(S0)
    d2 = similar(S0)

    fill!(S0,NaN)
    @views fill!(S0[1:r,1:r],zero(T))
    if flag == "tsi"

        x = LinRange(0,10π,Nx+1)[1:end-1]
        v = LinRange(-9,9,Nv+1)[1:end-1]

        α = 1e-3
        k = 1/5
        v0 = 2.4
        X0[:,1] .= @. 1 + (α*cos(k*x))
        V0[:,1] .= @. (exp(-.5*((v-v0)^2))+exp(-.5*((v+v0)^2)))/2/sqrt(2pi)
        S0[1,1] = 1.0

    elseif occursin("ll",flag)
        x = LinRange(0,4π,Nx+1)[1:end-1]
        v = LinRange(-6,6,Nv+1)[1:end-1]
        if flag == "nll"
            α = 5e-1
            k = 1/2
            X0[:,1] .= @. 1 + (α*cos(k*x))
            V0[:,1] .= @. (exp(-.5*((v)^2)))/sqrt(2pi)
            S0[1,1] = 1.0
        else
            α = 1e-2
            k = 1/2
            X0[:,1] .= @. 1 + (α*cos(k*x))
            V0[:,1] .= @. (exp(-.5*((v)^2)))/sqrt(2pi)
            S0[1,1] = 1.0
        end
    else
        error("Case not implemented")
    end
    kx = fftfreq(Nx)*(Nx)*2pi/(x[end]-x[1])
    kv = fftfreq(Nv)*(Nv)*2pi/(v[end]-v[1])
    hx = x[2]-x[1]
    hv = v[2]-v[1]

    Ax = diagm(1 => fill(1.0/(2.0*hx), Nx-1)) + diagm(-1 => fill(-1.0/(2.0*hx), Nx-1))
    Ax[1, end] = -1.0/(2.0*hx)
    Ax[end, 1] =  1.0/(2.0*hx)

    Av = diagm(1 => fill(1.0/(2.0*hv), Nv-1)) + diagm(-1 => fill(-1.0/(2.0*hv), Nv-1))
    Av[1, end] = -1.0/(2.0*hv)

    global Ax
    global Av

    tol = 1e-10

    if r > 1
        for i in 2:r
            X0[:,i] .= Pl.(cos.(x),i)
            V0[:,i] .= Pl.(sin.(v*pi/(-v[1])),i)
        end
    end
    X1,Rx1 = @views qr(X0[:,1:r])
    D = Diagonal(sign.(diag(Rx1)))
    @views mul!(X0[:,1:r],X1,D)
    Rx1 = D*Rx1
    V1,Rv1 = @views qr(V0[:,1:r])
    D = Diagonal(sign.(diag(Rv1)))
    @views mul!(V0[:,1:r],V1,D)
    Rv1 = D*Rv1
    @views S0[1:r,1:r] = Rx1*S0[1:r,1:r]*Rv1'
    tmp = rand(1,1)
    p = mnt(x=x,v=v,E=zeros(T,Nx),
            c1=tmp,c2=tmp,d1=tmp,d2=tmp,
            r=r,tmp=tmp,τ=τ,k=kx,h=hv,tol=tol,
            rx=0,rv=0,
            dx=(zeros(ComplexF64,Nx),kx,plan_fft(x)),
            dv=(zeros(ComplexF64,Nv),kv,plan_fft(v))
            )
    return X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p
end

function Basis_Update_step(A0,S0,B0,tmp,r,tspan,p,flag)
    p.tmp = @views @inbounds tmp[:,1:r]
    if flag == "K-step"
        @fastmath @views @inbounds mul!(B0[:,1:r],A0[:,1:r],S0[1:r,1:r])
        func = K_dot!
    elseif flag == "L-step"
        @fastmath @views @inbounds mul!(B0[:,1:r],A0[:,1:r],S0[1:r,1:r]')
        func = L_dot!
    else
        err("No such step exists.")
    end

    prob = @views @inbounds ODEProblem(func,B0[:,1:r],tspan,p);
    B1 = solve(prob,alg(),
                     reltol=1e-10,abstol=1e-10,
                    save_everystep=false,save_start=false,
                    save_end=true).u[end];
    @views @inbounds copy!(tmp[:,1:r], B1)
    @views @inbounds copy!(tmp[:,r+1:2*r], A0[:,1:r])
    rnew = @views @inbounds rank(tmp[:,1:2*r])
    A1 = @views @inbounds qr!(tmp[:,1:2*r]).Q[:,1:rnew] |> Matrix
    return A1,rnew
end


function BUG_step!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,augment)
    τ = p.τ
    p.r = r
    @views calc_E!(p.E,X0[:,1:r],S0[1:r,1:r],V0[:,1:r],p.h,p.dx);

    @views @inbounds calc_c1!(c1[1:r,1:r],p.v,V0[:,1:r])
    @views @inbounds calc_c2!(c2[1:r,1:r],p.dv,V0[:,1:r],tmpv[:,1:r])
    @views @inbounds calc_d1!(d1[1:r,1:r],p.E,X0[:,1:r])
    @views @inbounds calc_d2!(d2[1:r,1:r],p.dx,X0[:,1:r],tmpx[:,1:r])

    p.c1,p.c2 = @views @inbounds c1[1:r,1:r],c2[1:r,1:r]
    p.d1,p.d2 = @views @inbounds d1[1:r,1:r],d2[1:r,1:r]

    tspan = (0,τ);

    ######### BEGIN Basis-Update and Galerkin #############

    #### K-step ####
    X1,rx = Basis_Update_step(X0,S0,K0,tmpx,r,tspan,p,"K-step")

    #### L-step ####
    V1,rv = Basis_Update_step(V0,S0,L0,tmpv,r,tspan,p,"L-step")

    ### S-Step ####
    @views @inbounds S0[1:rx,1:rv] = @fastmath (X1' * X0[:,1:r]) * S0[1:r,1:r] * (V1' * V0[:,1:r])'

    ##### No need to recompute E!
    @views @inbounds calc_c1!(c1[1:rv,1:rv],p.v,V1)
    @views @inbounds calc_c2!(c2[1:rv,1:rv],p.dv,V1,tmpv[:,1:rv])
    @views @inbounds calc_d1!(d1[1:rx,1:rx],p.E,X1)
    @views @inbounds calc_d2!(d2[1:rx,1:rx],p.dx,X1,tmpx[:,1:rx])

    p.c1,p.c2 = @views @inbounds c1[1:rv,1:rv],c2[1:rv,1:rv]
    p.d1,p.d2 = @views @inbounds d1[1:rx,1:rx],d2[1:rx,1:rx]

    probS = @views @inbounds ODEProblem(S_dot!,S0[1:rx,1:rv],tspan,p);
    S1 = solve(probS,alg(),#dt=τ,adaptive=false,#dtmin=τ,force_dtmin=true,
                     reltol=1e-10,abstol=1e-10,
                    save_everystep=false,save_start=false,
                    save_end=true).u[end];

    if !isnothing(augment)
        P,Σ,Q = svd!(S1,alg=LinearAlgebra.QRIteration())
    #### Truncation
        if augment
            r1 = 0
            s = norm(Σ)
            tol = (p.tol*s)^2
            for i ∈ 2:min(rx,rv)
                if @views sum(Σ[i:end].^2) <= tol
                    r1 = i-1
                    break
                else
                    r1 = i
                end
            end
         #### Step rejection strategy
            if r1 >= rmax
                r1 = rmax
            elseif r1 == 2r
                r = r1
                @views copy!(X0[:,1:r],X1)
                @views copy!(V0[:,1:r],V1) ### S0 is already augmented but not updated
                BUG_step!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,augment)
                r1 = r
            end
            r = copy(r1)
            p.r = r
            p.rx = r
            p.rv = r
        elseif augment == false
            r = r
            p.r = r
            p.rx = r
            p.rv = r
        end
        @views @inbounds @fastmath mul!(X0[:,1:r],X1,P[:,1:r])
        @views @inbounds @fastmath mul!(V0[:,1:r],V1,Q[:,1:r])
        @views @inbounds copy!(S0[1:r,1:r],diagm(Σ[1:r]))
    else
        p.rx = rx
        p.rv = rv
        @views @inbounds copy!(X0[:,1:rx],X1)
        @views @inbounds copy!(V0[:,1:rv],V1)
        @views @inbounds copy!(S0[1:rx,1:rv],S1)
    end
    return nothing
end

function mBUG_step!(X0::AbstractArray,V0::AbstractArray,S0::AbstractArray,
                       K0::AbstractArray,L0::AbstractArray,
                       tmpx::AbstractArray,tmpv::AbstractArray,
                       c1::AbstractArray,c2::AbstractArray,
                       d1::AbstractArray,d2::AbstractArray,
                       p::MutableNamedTuple,r::Integer,rmax::Integer,
                       ic::AbstractArray,augment::Bool)
    r0 = copy(r)

    @views @inbounds copy!(ic[1][:,1:r],X0[:,1:r])
    @views @inbounds copy!(ic[2][:,1:r],V0[:,1:r])
    @views @inbounds copy!(ic[3][1:r,1:r],S0[1:r,1:r])

    X = @views @inbounds ic[1][:,1:r]
    V = @views @inbounds ic[2][:,1:r]
    S = @views @inbounds ic[3][1:r,1:r]

    p.τ /= 2
    BUG_step!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,nothing)
    p.τ *= 2
    rbar = p.rx + p.rv

    @views @inbounds calc_E!(p.E,X0[:,1:p.rx],S0[1:p.rx,1:p.rv],V0[:,1:p.rv],p.h,p.dx);
    @views @inbounds calc_d1!(p.d1,p.E,X0[:,1:p.rx])

    @views @inbounds @fastmath mul!(K0[:,1:p.rv],X0[:,1:p.rx],S0[1:p.rx,1:p.rv])
    p.tmp = @views @inbounds tmpx[:,1:p.rv]
    @views @inbounds K_dot!(tmpx[:,p.rx+1:rbar],K0[:,1:p.rv],p,0)
    @inbounds @views copy!(tmpx[:,1:p.rx],X0[:,1:p.rx])
    rx = @views @inbounds rank(tmpx[:,1:rbar])
    Xbar = @views @inbounds qr!(tmpx[:,1:rbar]).Q[:,1:rx]

    @views @inbounds @fastmath mul!(L0[:,1:p.rx],V0[:,1:p.rv],S0[1:p.rx,1:p.rv]')
    p.tmp = @views @inbounds tmpv[:,1:p.rx]
    @views @inbounds L_dot!(tmpv[:,p.rv+1:rbar],L0[:,1:p.rx],p,0)
    @views @inbounds copy!(tmpv[:,1:p.rv],V0[:,1:p.rv])
    rv = @views @inbounds rank(tmpv[:,1:rbar])
    Vbar = @views @inbounds qr!(tmpv[:,1:rbar]).Q[:,1:rv]

    Sbar = (Xbar' * X) * S * (Vbar' * V)'

    @views @inbounds calc_c1!(c1[1:rv,1:rv],p.v,Vbar)
    @views @inbounds calc_c2!(c2[1:rv,1:rv],p.dv,Vbar,tmpv[:,1:rv])
    @views @inbounds calc_d1!(d1[1:rx,1:rx],p.E,Xbar)
    @views @inbounds calc_d2!(d2[1:rx,1:rx],p.dx,Xbar,tmpx[:,1:rx])

    p.c1,p.c2 = @views @inbounds c1[1:rv,1:rv],c2[1:rv,1:rv]
    p.d1,p.d2 = @views @inbounds d1[1:rx,1:rx],d2[1:rx,1:rx]

    probS = ODEProblem(S_dot!,Sbar,(0,p.τ),p)
    S1 = solve(probS,alg(),#dt=p.τ,adaptive=false,
            reltol=1e-10,abstol=1e-10,
            save_start=false,save_everystep=false,save_end=true).u[end]

    P,Σ,Q = svd!(S1,alg=LinearAlgebra.QRIteration())

    if augment
        s = norm(Σ)
        tol = (p.tol*s)^2
        r1 = 0
        for i ∈ 2:min(rx,rv)
            if @views sum(Σ[i:end].^2) <= tol
                r1 = i-1
                break
            else
                r1 = i
            end
        end
        if r1 >= rmax
            r1 = rmax
        end
        r = r1
        p.r = r
    else
        r = r0
        p.r = r
    end
    @views @inbounds @fastmath mul!(X0[:,1:r],Xbar,P[:,1:r])
    @views @inbounds @fastmath mul!(V0[:,1:r],Vbar,Q[:,1:r])
    @views @inbounds copy!(S0[1:r,1:r],diagm(Σ[1:r]));
end

function get_sol(N,r,τ,T,tol,flag,integrator,augment)
    rmax = 2*r
    X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p = initialize(N,r,rmax,τ,flag,integrator)
    p.tol = tol
    ic = [similar(X0),similar(V0),similar(S0)]
    t = range(τ,T,step=τ)
    prog = Progress(length(t))
    for i in t
        if integrator == "mBUG"
            mBUG_step!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,ic,augment)
        else
            BUG_step!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,augment)
        end
        r = p.r

        next!(prog,showvalues = [(:Rank,r),(:Time,i)],valuecolor=:yellow)
    end

    f = X0[:,1:r]*S0[1:r,1:r]*V0[:,1:r]'

    writedlm("ref_sol.dat",f,",")
end

function der_fft!(dmat::AbstractArray,mat::AbstractMatrix,args)
    for (i,vec) in enumerate(eachcol(mat))
        dmat[:,i] .= der_fft(vec,args[1],args[2],args[3])
    end
    return dmat
end

function der_fft(mat::AbstractMatrix,p)
    dmat = similar(mat)
    for (i,vec) in enumerate(eachcol(mat))
        dmat[:,i] .= der_fft(vec,args[1],args[2],args[3])
    end
    return dmat
end

function der_fft(y::AbstractVector,Yk,k,plan)

    Yk .= plan*y
    lmul!(Diagonal(im*k),Yk)
    Yk = real.(plan\Yk)
    return Yk
end

function der_fft(y::AbstractVector,x::AbstractArray,)

    N = length(x)
    L = x[end]-x[1]
    k = fftfreq(N)*N*2pi/(L)
    Yk = fft(y)

    to_ifft = im*k.*Yk
    dy = real.(ifft(to_ifft))

    return dy
end


end
