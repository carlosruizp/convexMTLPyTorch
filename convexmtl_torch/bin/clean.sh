problem=$1


if [[ $problem = "all" ]]
then
    rm -f results/*
    rm -rf plots/*
    rm -rf logs/*
    rm -rf predictions/*
else
    rm -f results/*$problem*
    rm -rf plots/$problem
    rm -rf logs/$problem
    rm -rf predictions/*$problem*
fi

