using DLRA_Vlasov
using Documenter

DocMeta.setdocmeta!(DLRA_Vlasov, :DocTestSetup, :(using DLRA_Vlasov); recursive=true)

makedocs(;
    modules=[DLRA_Vlasov],
    authors="Giorgos Vretinaris",
    sitename="DLRA_Vlasov.jl",
    format=Documenter.HTML(;
        canonical="https://gvretina.github.io/DLRA_Vlasov.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gvretina/DLRA_Vlasov.jl",
    devbranch="main",
)
