using DLRA_Vlasov
using Documenter
using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "refs.bib"),style=:authoryear)
DocMeta.setdocmeta!(DLRA_Vlasov, :DocTestSetup, :(using DLRA_Vlasov); recursive=true)

makedocs(;
    modules=[DLRA_Vlasov],
    authors="Giorgos Vretinaris",
    sitename="DLRA_Vlasov.jl",
    format=Documenter.HTML(;
        prettyurls = true,
        canonical="https://gvretina.github.io/DLRA_Vlasov.jl",
        edit_link="master",
        assets=String["assets/citations.css"],
    ),
    pages=[
        "Home" => "index.md",
    ],
    plugins=[bib],
)

deploydocs(;
    repo="github.com/gvretina/DLRA_Vlasov.jl",
    devbranch="master",
)
