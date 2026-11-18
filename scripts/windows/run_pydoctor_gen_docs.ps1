## Supprisingly this DOES work for each of the projects and produces a semi-usable html output in their relative `doc/api/index.html` files

cd H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy

pydoctor `
    --project-name=neuropy `
    --project-version=0.2.5 `
    --project-url=https://github.com/CommanderPho/neuropy/ `
    --html-viewsource-base=https://github.com/CommanderPho/neuropy/tree/0.2.5 `
    --html-base-url=https://neuropy.readthedocs.io/en/latest/api `
    --html-output=../NeuroPy/docs/api `
    --docformat=plaintext `
    ./neuropy

cd H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers

pydoctor `
    --project-name=pyphocorehelpers `
    --project-version=0.4.6 `
    --project-url=https://github.com/CommanderPho/pyPhoCoreHelpers/ `
    --html-viewsource-base=https://github.com/CommanderPho/pyPhoCoreHelpers/tree/0.4.6 `
    --html-base-url=https://pyphocorehelpers.readthedocs.io/en/latest/api `
    --html-output=./docs/api `
    --docformat=plaintext `
    ./src/pyphocorehelpers
 
cd H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis

pydoctor `
    --project-name=pyphoplacecellanalysis `
    --project-version=0.3.6 `
    --project-url=https://github.com/CommanderPho/pyPhoPlaceCellAnalysis/ `
    --html-viewsource-base=https://github.com/CommanderPho/pyPhoPlaceCellAnalysis/tree/0.3.6 `
    --html-base-url=https://pyphoplacecellanalysis.readthedocs.io/en/latest/api `
    --html-output=./docs/api `
    --docformat=plaintext `
    ./src/pyphoplacecellanalysis



# (spike3d) PS H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D> .\scripts\windows\run_pydoctor_gen_docs.ps1       
# adding directory H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy
# 105/105 modules processed, 0 warnings
# writing html to ../NeuroPy/docs/api using pydoctor.templatewriter.writer.TemplateWriter
# starting ModuleIndexPage ...took 0.043053s
# starting ClassIndexPage ...took 0.073894s
# starting NameIndexPage ...took 0.118530s
# starting UndocumentedSummaryPage ...took 0.058944s
# starting HelpPage ...took 0.023597s
# starting AllDocuments ...took 1.520420s
# starting lunr search index ...took 1.384645s
# 274/274 pages written
# Generating objects inventory at ../NeuroPy/docs/api\objects.inv
# adding directory H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers
# 23/23 modules processed, 0 warnings
# writing html to ./docs/api using pydoctor.templatewriter.writer.TemplateWriter
# starting ModuleIndexPage ...took 0.076156s
# starting ClassIndexPage ...took 0.030346s
# starting NameIndexPage ...took 0.038753s
# starting UndocumentedSummaryPage ...took 0.022228s
# starting HelpPage ...took 0.031240s
# starting AllDocuments ...took 0.336897s
# starting lunr search index ...took 0.351607s
# 78/78 pages written
# Generating objects inventory at ./docs/api\objects.inv
# adding directory H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis
# 28/28 modules processed, 0 warnings
# writing html to ./docs/api using pydoctor.templatewriter.writer.TemplateWriter
# starting ModuleIndexPage ...took 0.014649s
# starting ClassIndexPage ...took 0.026064s
# starting NameIndexPage ...took 0.041092s
# starting UndocumentedSummaryPage ...took 0.016185s
# starting HelpPage ...took 0.018982s
# starting AllDocuments ...took 0.463988s
# starting lunr search index ...took 0.697800s
# 88/88 pages written
# Generating objects inventory at ./docs/api\objects.inv


# Open the generated documentation in the default browser
Write-Host "Opening generated documentation..." -ForegroundColor Green

# Open NeuroPy docs
$neuropyDocsPath = "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\docs\api\index.html"
if (Test-Path $neuropyDocsPath) {
    Start-Process $neuropyDocsPath
    Write-Host "Opened NeuroPy docs" -ForegroundColor Cyan
}

# Open pyPhoCoreHelpers docs
$pyphocorehelpersDocsPath = "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\docs\api\index.html"
if (Test-Path $pyphocorehelpersDocsPath) {
    Start-Process $pyphocorehelpersDocsPath
    Write-Host "Opened pyPhoCoreHelpers docs" -ForegroundColor Cyan
}

# Open pyPhoPlaceCellAnalysis docs
$pyphoplacecellanalysisDocsPath = "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\docs\api\index.html"
if (Test-Path $pyphoplacecellanalysisDocsPath) {
    Start-Process $pyphoplacecellanalysisDocsPath
    Write-Host "Opened pyPhoPlaceCellAnalysis docs" -ForegroundColor Cyan
}

Write-Host "Documentation generation complete!" -ForegroundColor Green

