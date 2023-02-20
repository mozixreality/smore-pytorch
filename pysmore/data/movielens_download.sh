#!/bin/sh

movielen_data="ml-1m"

if [ -d $movielen_data ]
then
    rm -rf $movielen_data $movielen_data.*
fi

wget https://files.grouplens.org/datasets/movielens/$movielen_data.zip
unzip -o $movielen_data.zip
rm $movielen_data.zip

awk -F '::' '{print "usr-"$1, "itm-"$2, $3}' $movielen_data/ratings.dat > net.dat