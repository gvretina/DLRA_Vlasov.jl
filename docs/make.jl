using DLRA_Vlasov
using Documenter

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))
DocMeta.setdocmeta!(DLRA_Vlasov, :DocTestSetup, :(using DLRA_Vlasov); recursive=true)

makedocs(;
    modules=[DLRA_Vlasov],
    authors="Giorgos Vretinaris",
    sitename="DLRA_Vlasov.jl",
    format=Documenter.HTML(;
        canonical="https://gvretina.github.io/DLRA_Vlasov.jl",
        edit_link="main",
        assets=String["assets/citations.css"],
    ),
    pages=[
        "Home" => "index.md",
    ],
    plugins=[bib],
)

deploydocs(;
    repo="github.com/gvretina/DLRA_Vlasov.jl",
    devbranch="main",
)
