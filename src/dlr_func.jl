using LinearAlgebra
# using MKL  ### use if FDM
using FFTW
using Tullio
using OrdinaryDiffEq
using MutableNamedTuples
using ProgressMeter
using GLMakie
using FStrings
using LaTeXStrings
using LegendrePolynomials
const mnt = MutableNamedTuple
# chebyshevt(n,x) = cos(n*acos(x))
alg() = Tsit5()#Euler()#AutoVern7(Rodas5P())


### Write it to encompass changes of potential
function RungeKutta4(RHS,y0,t0,h,p)
    k1 = h*RHS(y0,p,t0)
    k2 = h*RHS(y0+k1/2,p,t0+h/2)
    k3 = h*RHS(y0+k2/2,p,t0+h/2)
    k4 = h*RHS(y0+k3,p,t0+h)
    y1 = y0 + 1/6*(k1+2*k2+2*k3+k4)
    y1
end

function S_RK4(S0,X,V,E12,p)
    h = p.τ
    k1 = h*S_dot(S0,p,0)
    calc_d1!(p.d1,E12,X)
    k2 = h*S_dot(S0+k1/2,p,p.τ/2)
    k3 = h*S_dot(S0+k2/2,p,p.τ/2)
    calc_E!(p.E,X,S0+k3,V,p.k,p.h)
    calc_d1!(p.d1,p.E,X)
    k4 = h*S_dot(S0+k3,p,p.τ)

    S1 = S0 + 1/6*(k1+2*k2+2*k3+k4)
    S1
end

function simps_weights(n)
    w = 2*ones(Float64,n)/3
    w[1]=1/3
    w[2:2:n].=4/3
    if mod(n,3)!=0
        w[end-3] = 1/3+3/8
        w[end-2] = 9/8
        w[end-1] = 9/8
    end
    w[end]=3/8
    w
end

function trapz_weights(n)
    w = ones(Float64,n)
    w[1] = .5
    w[end] = .5
    w
end

################## 1D #################

function A_dot(Y0::AbstractArray,p,t)
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
    x = p.x
    v = p.v
    E = p.E
    dxX = der_fft(X0,x)
    dvV = der_fft(V0,v)

    return -dxX*S0*(v.*V0)' + (E .* X0)*S0*dvV'
end

function K_dot(K0,p,t)
    E = p.E
    c1 = p.c1
    c2 = p.c2
    x = p.x

#     dxK = Ax*K0
    dxK = der_fft(K0,x)
    K̇ = -dxK*c1' + (K0 .* E)*c2'
#     @tullio K̇[a,j] := -c1[j,l] * dxK[a,l] + c2[j,l] * E[a] * K0[a,l]
    return K̇
end

function K_dot!(K̇,K0,p,t)
    E = p.E
    c1 = p.c1
    c2 = p.c2
    x = p.x
    tmp = p.tmp

#     K̇ = -dxK*c1' + (K0 .* E)*c2'
#     K0E = E .* K0
#     @tullio K̇[a,j] = -c1[j,l] * dxK[a,l]
#     @tullio K̇[a,j] += c2[j,l] * K0E[a,l]

    dxK = der_fft(K0,x) ### IF_n *D_n * F_n
#     dxK = Ax*K0
    @fastmath mul!(K̇,-dxK,c1')
    @fastmath mul!(tmp,Diagonal(E),K0)
    @fastmath mul!(K̇,tmp,c2',1,1)
end

function L_dot(L0,p,t)
    v = p.v
    d1 = p.d1
    d2 = p.d2

    dvL = der_fft(L0,v)
#     dvL = Av*L0
    L̇ = -(L0 .* v)*d2' + dvL*d1'
#     @tullio L̇[a,i] := -d2[i,k] * v[a] * L0[a,k] + d1[i,k] * dvL[a,k]
    return L̇
end

function L_dot!(L̇,L0,p,t)
    v = p.v
    d1 = p.d1
    d2 = p.d2
    tmp = p.tmp

    dvL = der_fft(L0,v)
#     dvL = Av*L0

    @fastmath mul!(L̇,dvL,d1')
    @fastmath mul!(tmp,Diagonal(v),L0)
    @fastmath mul!(L̇,-tmp,d2',1,1)

end

function S_dot(S0,p,t)
    c1 = p.c1
    c2 = p.c2
    d1 = p.d1
    d2 = p.d2

    Ṡ = -d2*S0*c1' + d1*S0*c2'
#     @tullio Ṡ[i,j] := (-c1[j,l] * d2[i,k] + c2[j,l] * d1[i,k]) * S0[k,l]
    return Ṡ
end

function S_dot!(Ṡ,S0,p,t)
    c1 = p.c1
    c2 = p.c2
    d1 = p.d1
    d2 = p.d2

#     Ṡ = -d1*S0*c1' + d2*S0*c2'
    @tullio Ṡ[i,j] = (-c1[j,l] * d2[i,k] + c2[j,l] * d1[i,k]) * S0[k,l]
end

function calc_c(v,V0,r)#,w,h)
    dvV = der_fft(V0,v)

    c1 = V0' * (v .* V0)# * h
    c2 = V0' * (dvV)# * h
#     vV0 = v .* V0
#     @tullio c1[j,l] := V0[a,j]*vV0[a,l]#*w[a]*h
#     @tullio c2[j,l] := V0[a,j]*dvV[a,l]#*w[a]*h

    return SMatrix{r,r}(c1),SMatrix{r,r}(c2)
end


function calc_c1!(c1,v,V0)
    @fastmath mul!(c1,V0', v.*V0)
end


function calc_c2!(c2,v,V0)
#     dvV = Av*V0
    dvV = der_fft(V0,v)
    @fastmath mul!(c2,V0',dvV)
end

function calc_d(x,X0,E,r)#,w,h)

    dxX = der_fft(X0,x)

    d1 = X0' * (E .* X0)# * h
    d2 = X0' * dxX# * h
#     @tullio d1[i,k] := X0[a,i]*E[a]*X0[a,k]#*w[a]*h
#     @tullio d2[i,k] := X0[a,i]*dxX[a,k]#*w[a]*h
    return SMatrix{r,r}(d1),SMatrix{r,r}(d2)
end

function calc_d1!(d1,E,X0)
    @fastmath mul!(d1, X0', E .* X0)
end

function calc_d2!(d2,x,X0)
#     dxX = Ax*X0
    dxX = der_fft(X0,x)
    @fastmath mul!(d2, X0', dxX)
end

function calc_E(X0,S0,V0,k,h)
    ρ = X0*S0*sum(V0,dims=1)'*h
    ρ̂ = @views fft(1 .- ρ)#[1:end-1])
    Ê = -im * ρ̂ ./ k
    Ê[1] = zero(eltype(Ê))
    E = real.(ifft(Ê))
    return E#[E; E[1]]
end


function calc_E!(E,X0,S0,V0,k,h)
    ρ̂ = Vector{ComplexF64}(undef,k.n)
    ρ̂ .= 1 .- X0*S0*sum(V0,dims=1)'*h
    fft!(ρ̂)
    Ê = -im * ρ̂ ./ k
    Ê[1] = zero(eltype(Ê))
    E .= real.(ifft(Ê))

#     E[1:end-1] .= real.(ifft(Ê))
#     E[end] = E[1]
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
    K0 = similar(X0)#Array{T}(undef,Nx,2*rmax)
    L0 = similar(V0)#Array{T}(undef,Nv,2*rmax)
    tmpx = similar(X0)#Array{T}(undef,Nx,2*rmax)
    tmpv = similar(V0)#Array{T}(undef,Nv,2*rmax)
    c1 = similar(S0)#Array{T}(undef,2*rmax,2*rmax)
    c2 = similar(S0)#Array{T}(undef,2*rmax,2*rmax)
    d1 = similar(S0)#Array{T}(undef,2*rmax,2*rmax)
    d2 = similar(S0)#Array{T}(undef,2*rmax,2*rmax)

    fill!(S0,NaN)
    @views fill!(S0[1:r,1:r],zero(T))
    if flag == "tsi"

        x = LinRange(0,10π,Nx+1)[1:end-1]
        v = LinRange(-9,9,Nv+1)[1:end-1]
        kx = fftfreq(Nx)*(Nx)*2pi/(x[end]-x[1])
        hx = x[2]-x[1]
        hv = v[2]-v[1]

        α = 1e-3
        k = 1/5
        v0 = 2.4
        X0[:,1] .= @. 1 + (α*cos(k*x))
        V0[:,1] .= @. (exp(-.5*((v-v0)^2))+exp(-.5*((v+v0)^2)))/2/sqrt(2pi)
        S0[1,1] = 1.0

    elseif occursin("ll",flag)
        x = LinRange(0,4π,Nx+1)[1:end-1]
        v = LinRange(-6,6,Nv+1)[1:end-1]
        kx = fftfreq(Nx)*(Nx)*2pi/(x[end]-x[1])
        hx = x[2]-x[1]
        hv = v[2]-v[1]
        if flag == "nll"
            α = 5e-1
            k = 1/2
            X0[:,1] .= @. 1 + (α*cos(k*x))
            V0[:,1] .= @. (exp(-.5*((v)^2)))/sqrt(2pi)
            S0[1,1] = 1.0
        else
            α = 1#e-2
            k = 1/2
            X0[:,1] .= @. 1 + (α*cos(k*x))
            V0[:,1] .= @. (exp(-.5*((v)^2)))/sqrt(2pi)
            S0[1,1] = 1.0
        end
    else
        error("Case not implemented")
    end

    Ax = diagm(1 => fill(1.0/(2.0*hx), Nx-1)) + diagm(-1 => fill(-1.0/(2.0*hx), Nx-1))
    Ax[1, end] = -1.0/(2.0*hx)
    Ax[end, 1] =  1.0/(2.0*hx)

    Av = diagm(1 => fill(1.0/(2.0*hv), Nv-1)) + diagm(-1 => fill(-1.0/(2.0*hv), Nv-1))
    Av[1, end] = -1.0/(2.0*hv)

    global Ax
    global Av

    tol = 1e-10
#     α = 1e-3
#     k = 1/5
#     v0 = 2.4
#     X0[:,1] .= @. 1 + (α*cos(k*x))
# #     V0[:,1] .= @. (exp(-.5*((v)^2)))/sqrt(2pi)
#     V0[:,1] .= @. (exp(-.5*((v-v0)^2))+exp(-.5*((v+v0)^2)))/2/sqrt(2pi)
#     S0[1,1] = 1.0
    if r > 1
        for i in 2:r
            X0[:,i] .= Pl.(cos.(x),i)#chebyshevt.(i,range(-1,1,Nx))
            V0[:,i] .= Pl.(sin.(v*pi/(-v[1])),i)#chebyshevt.(i,range(-1,1,Nv))
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
    p = mnt(x=x,v=v,E=zeros(T,Nx),#Vector{T}(undef,Nx),
            c1=tmp,c2=tmp,d1=tmp,d2=tmp,
            r=r,tmp=tmp,τ=τ,k=kx,h=hv,tol=tol,
            rx=0,rv=0)
    return X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p
end

function get_correct_QR(Y,r)
    Q,R = qr!(Y)
    r_new = rank(R)
    idx_smallest = partialsortperm(abs.(diag(R)), 1:(2r-r_new))
    mask = BitVector(undef,2r)
    mask .= true #[true for i in 1:2r]
    mask[idx_smallest] .= false
    return @inbounds Matrix(Q)[:,mask], r_new
end

function BUG_step!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,augment)
    τ = p.τ
    p.r = r
#     @views calc_E!(p.E,X0[:,1:r],S0[1:r,1:r],V0[:,1:r],p.k,p.h);

    @views @inbounds calc_c1!(c1[1:r,1:r],p.v,V0[:,1:r])
    @views @inbounds calc_c2!(c2[1:r,1:r],p.v,V0[:,1:r])
    @views @inbounds calc_d1!(d1[1:r,1:r],p.E,X0[:,1:r])
    @views @inbounds calc_d2!(d2[1:r,1:r],p.x,X0[:,1:r])

    p.c1,p.c2 = @views @inbounds c1[1:r,1:r],c2[1:r,1:r]
    p.d1,p.d2 = @views @inbounds d1[1:r,1:r],d2[1:r,1:r]
    p.tmp = @views @inbounds tmpx[:,1:r]

#     println("BUG time-step ",τ)
    tspan = (0,τ);

    ######### BEGIN Basis-Update and Galerkin #############

    #### K-step ####
    @fastmath @views @inbounds mul!(K0[:,1:r],X0[:,1:r],S0[1:r,1:r])
    probK = @views @inbounds ODEProblem(K_dot!,K0[:,1:r],tspan,p);
    K1 = solve(probK,alg(),dt=τ,adaptive=false,#dtmin=τ,force_dtmin=true,
#                      reltol=1e-10,abstol=1e-10,
                    save_everystep=false,save_start=false,
                    save_end=true).u[end];
#     X1,rx = get_correct_QR([K1 X0[:,1:r]],r)
    X1,R = @views @inbounds qr!([K1 X0[:,1:r]])#.Q |> Matrix
#     mask = abs.(diag(R)) .> 10*eps()*norm(R)
#     rx = sum(mask)
#     println(rank(R)," ",rx)
#     @myshow diag(R)'
#     @assert rx == rank(R)
#     X1 = Matrix(X1)[:,mask]
    rx = rank(R)
    X1 = Matrix(X1)[:,1:rx]

    #### L-step ####
    p.tmp = @views @inbounds tmpv[:,1:r]
    @fastmath @views @inbounds mul!(L0[:,1:r],V0[:,1:r],S0[1:r,1:r]')
    probL = @views @inbounds ODEProblem(L_dot!,L0[:,1:r],tspan,p);
    L1 = solve(probL,alg(),dt=τ,adaptive=false,#dtmin=τ,force_dtmin=true,
#                      reltol=1e-10,abstol=1e-10,
                    save_everystep=false,save_start=false,
                    save_end=true).u[end];

#     V1,rv = get_correct_QR([L1 V0[:,1:r]],r)
    V1,R = @views @inbounds qr!([L1 V0[:,1:r]])#.Q |> Matrix
#     mask = abs.(diag(R)) .> 10*eps()*norm(R)
#     rv = sum(mask)
#     println(rank(R)," ",rv)
#     @myshow diag(R)'
#     @assert rv == rank(R)
#     V1 = Matrix(V1)[:,mask]
    rv = rank(R)#sum(mask)
    V1 = Matrix(V1)[:,1:rv]#mask]

    ### Update basis and Galerkin (S-Step) ####

#     p.r = 2*r
    @views @inbounds S0[1:rx,1:rv] = @fastmath (X1' * X0[:,1:r]) * S0[1:r,1:r] * (V1' * V0[:,1:r])'

#         @views calc_E!(p.E,X1,S0[1:p.r,1:p.r],V1,kx,hv);
    @views @inbounds calc_c1!(c1[1:rv,1:rv],p.v,V1)
    @views @inbounds calc_c2!(c2[1:rv,1:rv],p.v,V1)
    @views @inbounds calc_d1!(d1[1:rx,1:rx],p.E,X1)
    @views @inbounds calc_d2!(d2[1:rx,1:rx],p.x,X1)

    p.c1,p.c2 = @views @inbounds c1[1:rv,1:rv],c2[1:rv,1:rv]
    p.d1,p.d2 = @views @inbounds d1[1:rx,1:rx],d2[1:rx,1:rx]

    probS = @views @inbounds ODEProblem(S_dot!,S0[1:rx,1:rv],tspan,p);
    S1 = solve(probS,alg(),dt=τ,adaptive=false,#dtmin=τ,force_dtmin=true,
#                      reltol=1e-10,abstol=1e-10,
                    save_everystep=false,save_start=false,
                    save_end=true).u[end];
    if isnothing(augment) == false
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
        #         #### Step rejection strategy
            if r1 >= rmax
                r1 = rmax
            elseif r1 == 2r
                r = r1
                @views copy!(X0[:,1:r],X1)
                @views copy!(V0[:,1:r],V1) ### S0 is already augmented but not updated
                BUG_step!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,augment)
            end
            r = copy(r1)
            p.r = r
            p.rx = r
            p.rv = r
        elseif augment == false
            r = r#p.r÷2
            p.r = r
            p.rx = r
            p.rv = r
        end
        #### Keep copy of values after one time step ####
        try
            @views @inbounds @fastmath mul!(X0[:,1:r],X1,P[:,1:r])
            @views @inbounds @fastmath mul!(V0[:,1:r],V1,Q[:,1:r])
            @views @inbounds copy!(S0[1:r,1:r],diagm(Σ[1:r]))
        catch err
            @myshow r
            @myshow X1
            @myshow P[:,1:r]
            err
        end
    else
        p.rx = rx
        p.rv = rv
        @views @inbounds copy!(X0[:,1:rx],X1)
        @views @inbounds copy!(V0[:,1:rv],V1)
        @views @inbounds copy!(S0[1:rx,1:rv],S1)
    end
    return nothing
end

function midpoint_BUG!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,ic,augment)
    r0 = copy(r)
    @views @inbounds copy!(ic[1][:,1:r],X0)
    @views @inbounds copy!(ic[2][:,1:r],V0)
    @views @inbounds copy!(ic[3][1:r,1:r],S0)

    X = @views @inbounds ic[1][:,1:r]
    V = @views @inbounds ic[2][:,1:r]
    S = @views @inbounds ic[3][1:r,1:r]
#     X = copy(X0)
#     V = copy(V0)
#     S = copy(S0)


    p.τ = p.τ/2
#     println("BUG rank in ",r)
    BUG_step!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,nothing)#trunc(rmax÷2))
    r = p.r

    p.τ = p.τ*2
#     println("BUG rank out ",r)
    ### t = t0 + h/2
#     @views @inbounds calc_E!(p.E,X0[:,1:r],S0[1:r,1:r],V0[:,1:r],p.k,p.h);
    E12 = copy(p.E)
    @views @inbounds calc_c1!(c1[1:p.rv,1:p.rv],p.v,V0[:,1:p.rv])
    @views @inbounds calc_c2!(c2[1:p.rv,1:p.rv],p.v,V0[:,1:p.rv])
    @views @inbounds calc_d1!(d1[1:p.rx,1:p.rx],p.E,X0[:,1:p.rx])
    @views @inbounds calc_d2!(d2[1:p.rx,1:p.rx],p.x,X0[:,1:p.rx])

    p.c1,p.c2 = @views @inbounds c1[1:p.rv,1:p.rv],c2[1:p.rv,1:p.rv]
    p.d1,p.d2 = @views @inbounds d1[1:p.rx,1:p.rx],d2[1:p.rx,1:p.rx]

    @views @inbounds @fastmath mul!(K0[:,1:p.rv],X0[:,1:p.rx],S0[1:p.rx,1:p.rv])
    K1 = @views @inbounds K_dot(K0[:,1:p.rv],p,0)
#     K1 = @views @inbounds A_dot(X0[:,1:r],S0[1:r,1:r],V0[:,1:r],p,0)*V0[:,1:r]
    Xbar,R = @views @inbounds qr!([X0[:,1:p.rx] K1])#.Q |> Matrix
    rx = rank(R)
    Xbar = Matrix(Xbar)[:,1:rx]
    @views @inbounds @fastmath mul!(L0[:,1:p.rx],V0[:,1:p.rv],S0[1:p.rx,1:p.rv]')
    L1 = @views L_dot(L0[:,1:p.rx],p,0)
#     L1 = @views @inbounds A_dot(X0[:,1:r],S0[1:r,1:r],V0[:,1:r],p,0)'*X0[:,1:r]
    Vbar,R = @views @inbounds qr!([V0[:,1:p.rv] L1])#.Q |> Matrix
    rv = rank(R)
    Vbar = Matrix(Vbar)[:,1:rv]

    ### t = t0
#     p.r = 2*r
    Sbar = @views @inbounds @fastmath (Xbar' * X[:,1:r0]) * S[1:r0,1:r0] * (Vbar' * V[:,1:r0])'

#     calc_E!(p.E,Xbar,Sbar,Vbar,p.k,p.h)
#     @views @inbounds calc_E!(p.E,ic[1][:,1:r0],ic[3][1:r0,1:r0],ic[2][:,1:r0],p.k,p.h)
    @views @inbounds calc_c1!(c1[1:rv,1:rv],p.v,Vbar)
    @views @inbounds calc_c2!(c2[1:rv,1:rv],p.v,Vbar)
    @views @inbounds calc_d1!(d1[1:rx,1:rx],p.E,Xbar)
    @views @inbounds calc_d2!(d2[1:rx,1:rx],p.x,Xbar)
    p.c1,p.c2 = @views @inbounds c1[1:rv,1:rv],c2[1:rv,1:rv]
    p.d1,p.d2 = @views @inbounds d1[1:rx,1:rx],d2[1:rx,1:rx]

#     println("mBUG rank in ",r)
#     println("mBUG time-step ",p.τ)

    probS = ODEProblem(S_dot!,Sbar,(0,p.τ),p)
    S1 = solve(probS,alg(),#dt=p.τ,adaptive=false,
               reltol=1e-10,abstol=1e-10,
               save_start=false,save_everystep=false,save_end=true).u[end]
#     S1 = S_RK4(Sbar,Xbar,Vbar,E12,p)
    P,Σ,Q = svd!(S1,alg=LinearAlgebra.QRIteration())

    #### Truncation
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
        #         #### Step rejection strategy
        if r1 >= rmax
            r1 = rmax
        end
        r = r1
        p.r = r
    else
        r = r0#p.r ÷ 2
#         r = min(r0,p.r ÷ 2)
        p.r = r
    end
#     r = r0
#     p.r = r
#     println("mBUG rank out ",r)

    #### Keep copy of values after one time step ####s

    @views @inbounds @fastmath mul!(X0[:,1:r],Xbar,P[:,1:r])
    @views @inbounds @fastmath mul!(V0[:,1:r],Vbar,Q[:,1:r])
    @views @inbounds copy!(S0[1:r,1:r],diagm(Σ[1:r]));
    return nothing
end

function get_sol(N,r,τ,T,tol,flag,integrator,augment)
    rmax = 2*r
    X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p = initialize(N,r,rmax,τ,flag,integrator)
    p.tol = tol
    ic = nothing
    t = range(τ,T,step=τ)
    prog = Progress(length(t))
    for i in t
        if integrator == "mBUG"
            midpoint_BUG!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,ic,augment)
        else
            BUG_step!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,augment)
        end
        r = p.r

        next!(prog,showvalues = [(:Rank,r),(:Time,i)],valuecolor=:yellow)
    end

    f = X0[:,1:r]*S0[1:r,1:r]*V0[:,1:r]'

    writedlm("ref_sol.dat",f,",")
end

function errors(flag,tol,rmax)
    N = 2*128
    Nx = N#÷2
    Nv = N#*2

    augment = false
    set_theme!(theme_latexfonts())
    T = Float64
    f = Array{T}(undef,Nx,Nv)
    fill!(f,zero(T))

#     ref_sol = readdlm("ref_sol.dat",',',Float64)[1:4:end,1:4:end]
    x = LinRange(0,4π,Nx+1)[1:end-1]
    v = LinRange(-6,6,Nv+1)[1:end-1]
    ref_sol = advection(10,x,v)

    τs = .1 .* [.5^i for i in 0:7]
#     τs = [τs; 1e-4]
    f_list = [similar(f) for i in 0:7]
    r_list = [3,5,10,15]
    r = 0
    for integrator in ["BUG"]#,"mBUG"]
        fig = Figure()
        ax = Axis(fig[1,1],xlabel="time-step",ylabel="Error",xscale=log10,yscale=log10)
        ylims!(ax,1e-7,1e-1)
        display(fig)

        for r0 in reverse(r_list)
            if augment
                rmax = 4*r0
            else
                rmax = 2*r0
            end
            for (j,τ) in enumerate(τs)
#                 if τ == τs[end] && r0 != r_list[end]
#                     continue
#                 end
#                 t = range(τ,10,step=τ)
#                 n = length(t)
                n = 10 ÷ τ |> Int64 #trunc(Int64,10 / τ)
                if mod(n,10) == 9
                    n = n+1
                end
                r = r0
                X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p = initialize(N,r,rmax,τ,flag,integrator)
                @assert maximum(abs.(advection(0,p.x,p.v)-X0[:,1:r]*S0[1:r,1:r]*V0[:,1:r]')) <1e-15
                @assert p.x == x
                @assert p.v == v
                @assert n*τ == 10
                ic = [similar(X0),similar(V0),similar(S0)]
                p.tol = tol
                p.τ = τ
                prog = Progress(n)
                for i in 1:n
                    if integrator == "mBUG"
                        midpoint_BUG!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,ic,augment)
                    else
                        BUG_step!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,augment)
                    end
                    r = p.r

                    if norm(S0[1:r,1:r]) > 500
                        println("CFL condition probably not satisfied!")
                        finish!(prog)
                        break
                    end

                    next!(prog,showvalues = [(:Rank,r)
                                             (:Time,p.τ*i)
                                             (:τ,p.τ)
                                            (:E,norm(p.E))],valuecolor=:yellow)

                end

                @views mul!(tmpx[:,1:r],X0[:,1:r],S0[1:r,1:r])
                @views mul!(f,tmpx[:,1:r],V0[:,1:r]')
                copy!(f_list[j],f)
            end

#             copy!(ref_sol,f_list[end])

            d = [f_list[i]-ref_sol for i in 1:7]#diff(f_list)
            errF =  norm.(d)./norm(ref_sol) .+ eps()
            if r == r_list[end]
                lin = @. τs[1:end-1]/τs[1]*errF[end-2]
                quad = lin.^2 / errF[end-2]
                lines!(ax,τs[1:end-1],lin,color=:red,label="Linear")
                lines!(ax,τs[1:end-1],quad,color=:magenta,label="Quadratic")
            end
            scatterlines!(ax,τs[1:end-1],errF,label="Rank $(r)")

        end

        axislegend(ax,merge=true,position=:rb)
        save("figs/error_advection_$(integrator)_.png",fig)
#         save("figs/error_$(flag)_$(integrator)_num_ref_sol_mBUG_adaptive_518.png",fig)
    end
end

function anim(flag,r,tol,augment,integrator)
    N,τ,T = 128,1e-2,10
    n = div(T,τ)+1 |> Int64

    rmax = 2*r
    Nx = N#÷2
    Nv = N#*2

    X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p = initialize(N,r,rmax,τ,flag,integrator)
    p.tol = tol

    set_theme!(theme_latexfonts())
    fig = Figure(size=(1920,1080))
    ax1 = Axis3(fig[1,1],xlabel="x",ylabel="v",title="Numerical")
#     ax2 = Axis(fig[1,2],yscale=log10,title="Energy")
    ax2 = Axis3(fig[1,2],xlabel="x",ylabel="v",title="Analytical")
    lab = Label(fig[0,:], "t=0s")
    T = Float64
    f = Array{T}(undef,Nx,Nv)
    @views mul!(tmpx[:,1:r],X0[:,1:r],S0[1:r,1:r])
    @views mul!(f,tmpx[:,1:r],V0[:,1:r]')
#     en = Array{T}(undef,1)
#     @views calc_E!(p.E,X0[:,1:r],S0[1:r,1:r],V0[:,1:r],p.k,p.h);
#     mul!(en,p.E',p.E)
#     lmul!(p.x[2],en)
#     en1 = copy(en[1])
#     ax1.title = "0s"

    plot_y = Observable(f)
    plot_sol = Observable(advection(0,p.x,p.v))
    @assert maximum(abs.(f-plot_sol[])) < 1e-15

#     plot_E = Observable(Point2f[(0,en1)])
#     plot_a = Observable(Point2f[(0,en1*0.725)])

    surface!(ax1,p.x,p.v,plot_y,colormap=:magma)
    surface!(ax2,p.x,p.v,plot_sol,colormap=:magma)
#     lines!(ax2,plot_E)
#     if flag == "ll"
#         lines!(ax2,plot_a,color=:red)
#     end
    display(fig)
    p.τ = τ
    ic = [similar(X0),similar(V0),similar(S0)]

    prog = Progress(n)
    name = f"vids/{integrator}_{uppercase(flag)}_r_{rmax}_tol_{p.tol:.0e}.mkv"
    display(name)
#     record(fig, name, 1:n;
#            framerate = 80) do i
    for i in 1:n
        sleep(1/100)
        isopen(fig.scene) || break
        if integrator == "mBUG"
            midpoint_BUG!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,ic,augment)
        else
            BUG_step!(X0,V0,S0,K0,L0,tmpx,tmpv,c1,c2,d1,d2,p,r,rmax,augment)
        end
        r = p.r
        @views mul!(tmpx[:,1:r],X0[:,1:r],S0[1:r,1:r])
        @views mul!(f,tmpx[:,1:r],V0[:,1:r]')
#         @views calc_E!(p.E,X0[:,1:r],S0[1:r,1:r],V0[:,1:r],p.k,p.h);
#         mul!(en,p.E',p.E)
#         lmul!(p.x[2],en)

        #### for advection only
#         fill!(p.E,zero(eltype(p.E)))

        ti = τ*i
#         f .= X0[:,1:r]*S0[1:r,1:r]*V0[:,1:r]'
        plot_y[] = f
        plot_sol[] = (f-advection(p.τ*(i),p.x,p.v))
        lab.text = f"t={ti:.4f}s"
#         point = Point2f(ti,en[1])
#         plot_E[] = push!(plot_E[],point)
#         point = Point2f(ti,exp(-2*0.153*ti)*en1*0.725)
#         plot_a[] = push!(plot_a[],point)
#         ax1.title = f"t={ti:.4f}s  r={r}"
        autolimits!(ax2)
#         recordframe!(io)

#         if norm(S0[1:r,1:r]) > 500
#             finish!(prog)
#             break
#         end

        next!(prog,showvalues = [(:Rank,r)
                                 (:E,norm(p.E))
                                 (:τ,p.τ)],valuecolor=:yellow)
#     end
    end

end

function create_anims()
    flags = ["ll","nll","tsi"]
    tols = [1e-10]
    rs = [3,5,10]
    for i in Iterators.product(flags,rs,tols)
        anim(i[1],i[2],i[3],true,"mBUG")
    end
end

function der_fdm(y,x)
    h = x[2]-x[1]
    n = length(y)
    dy = zeros(eltype(y),n)
    for i in 1:n
        if i > 1 && i < n
            dy[i] = (y[i+1]-y[i-1])/(2h)
        elseif i == 1
            dy[i] = (-y[i+2]+4*y[i+1]-3*y[i])/(2h)
        else
            dy[i] = (3*y[i]-4*y[i-1]+y[i-2])/(2h)
        end
    end
    dy
end

function der_fdm(mat::AbstractMatrix,x::AbstractArray)
    dmat = similar(mat)
    for (i,vec) in enumerate(eachcol(mat))
        dmat[:,i] .= der_fdm(vec,x)
    end
    return dmat
end

function der_fdm(y::AbstractVector,x)
    h = x[2]-x[1]
    n = length(y)
    dy = similar(y)#zeros(eltype(y),n)
    for i in 1:n
        if i > 1 && i < n
            dy[i] = (y[i+1]-y[i-1])/(2h)
        elseif i == 1
            dy[i] = (-y[i+2]+4*y[i+1]-3*y[i])/(2h)
        else
            dy[i] = (3*y[i]-4*y[i-1]+y[i-2])/(2h)
        end
    end
    dy
end

function der_fft(mat::AbstractMatrix,x::AbstractArray)
    dmat = similar(mat)
    for (i,vec) in enumerate(eachcol(mat))
        dmat[:,i] .= der_fft(vec,x)
    end
    return dmat
end


function der_fft(y::AbstractVector,x::AbstractArray)

    N = length(x)
    L = x[end]-x[1]
    k = fftfreq(N)*N*2pi/(L)
    Yk = fft(y)

    to_ifft = im*k.*Yk
    dy = real.(ifft(to_ifft))
#     [dy[mask]; dy[1]]
    return dy
end

function der_fft(f::Function,x::AbstractArray)

    dx = x[2]-x[1]
    N = length(x)
    if x[end]+x[1] > 1e-4 && (f(x[end])-f(x[1]))^2 > 1e-4
        println("yes")
        a = x[end]#+dx
        xx = collect(range(-a,a,step=dx))[1:end-1]
        N = 2N-2
        L = 2a
        y = zeros(N)
        for i in 1:N
            if i <= N÷2
                y[i] = f(xx[i])
            else
                y[i] = f(xx[i])
            end
        mask = N÷2+1:1:N
        end
    elseif x[end]==-x[1]
        println("why")
        xx = x[1:end-1]
        N = N-1
        y = f.(xx)
        L = x[end]-x[1]
        mask = 1:N
    else
        xx = x[1:end-1]
        N = N-1
        y = f.(xx)
        L = x[end]
        mask = 1:N
    end
    k = fftfreq(N)*N*2pi/(L)
    Yk = fft(y)

    to_ifft = im*k.*Yk
    dy = real.(ifft(to_ifft))

    # fig,ax,_ = plot(xx,y)
    # lines!(xx,dy)
    # display(fig)
    # display([xx[mask]; a])
    [dy[mask]; dy[1]]
end

function advection(t,x,v)
    sol = Array{Float64}(undef,length(x),length(v))
    for (j,v) in enumerate(v)
        for (i,x) in enumerate(x)
            sol[i,j] = (1+cos(.5*(x-v*t)))*(exp(-.5*((v)^2)))/sqrt(2pi)
        end
    end
    return sol
end

macro myshow(exs...)
    blk = Expr(:block)
    for ex in exs
        push!(blk.args, :(print($(sprint(Base.show_unquoted,ex)*" = "))))
        push!(blk.args, :(show(stdout, "text/plain", begin value=$(esc(ex)) end)))
        push!(blk.args, :(println()))
    end
    isempty(exs) || push!(blk.args, :value)
    return blk
end
