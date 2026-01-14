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
    --project-version=0.3.5 `
    --project-url=https://github.com/CommanderPho/pyPhoPlaceCellAnalysis/ `
    --html-viewsource-base=https://github.com/CommanderPho/pyPhoPlaceCellAnalysis/tree/0.3.5 `
    --html-base-url=https://pyphoplacecellanalysis.readthedocs.io/en/latest/api `
    --html-output=./docs/api `
    --docformat=plaintext `
    ./src/pyphoplacecellanalysis

