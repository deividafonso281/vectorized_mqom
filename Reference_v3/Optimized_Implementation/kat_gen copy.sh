echo "👉 Gerando kats :)"

DATA=$(date '+%Y%m%d')

for dir in */; do
        echo "👉 Processando $dir"
        cd "$dir"

        make clean

        make kat_check

        ./kat_check

        make clean

        
        echo "👉 Repositorio limpo :)"

        cd ..
done
