#!/bin/sh

username=`whoami`

cp $1.dot $1-backup.dot
sed -i '/CSC2CBLK/d' $1.dot

max=`grep -v "\->" $1.dot | grep GEMM | sed 's/GEMM_[0-9]*_\([0-9]*\) \[.*/\1/' | sort -g | tail -n 1`
grep "^GEMM_[0-9]*_[0-9]* ->" $1.dot | sed 's/\(GEMM_[0-9]*_[0-9]* \)->.*/\1/' > /tmp/tmp-$username-sparse.log

echo $max

for i in `seq 0 $max`
do
    res=`grep "GEMM_[0-9]*_$i " /tmp/tmp-$username-sparse.log`
    #echo "Resultat is x${res}x"
    if [ -z "$res" ]
    then
        echo -n "\r$i"
        sed -i "/GEMM_[0-9]*_$i /d" $1.dot
    fi
done
echo "\n"

rm -f /tmp/tmp-$username-sparse.log
wc -l $1.dot