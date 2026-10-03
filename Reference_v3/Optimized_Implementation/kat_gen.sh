echo "👉 Gerando kats :)"

DATA=$(date '+%Y%m%d')

for dir in */; do
        echo "👉 Processando $dir"
        cd "$dir"

        make clean

        make kat_gen

        ./kat_gen

        make clean

        
        echo "👉 Repositorio limpo :)"

        cd ..
done
