for variant in reloaded.*; do

for dir in Korrigierte_Layout_GT/*/; do test $dir = ${dir/.} || continue; out=${dir%/}.eynollah-206-noautosize; mkdir -p $out; nohup /usr/bin
/time eynollah  -D GPU0 -m $variant layout --dir_in $dir -o
 $out -fl -H 1 -O || break; done

logfile=Korrigierte_Layout_GT/nohup.out.eynollah-206-noautosize-$variant

mv nohup.out $logfile

fgrep peaked $logfile | sort -u | cut -d\  -f2,5 > $logfile.peak-vram.txt
fgrep initialization $logfile | sed "s/^.* (\([0-9.]*\)s)/\1/" > $logfile.secs-init.txt
fgrep "Job done in" $logfile | sed "s/^.* //;s/s$//" > $logfile.secs-jobs.txt
fgrep %CPU $logfile | sed "s/^.* \([0-9]*\)%CPU.*/\1/" > $logfile.prct-cpu.txt

for dir in Korrigierte_Layout_GT/*.eynollah*/; do
	test $dir = ${dir/-eval} || continue;
	test $dir != ${dir/eynollah-206} || continue;
	mets=${dir%.eynollah*}.mets.xml;
	dir=$(basename $dir);
	gt=${dir%.eynollah*}.gt2;
	echo $mets: $dir;
	ocrd-segment-evaluate -m $mets -I $gt,$dir -O $dir-eval,$dir-eval2 -P level-of-operation region -P ignore-subtype true -P only-fg true --overwrite;
done

for mets in Korrigierte_Layout_GT/*.mets.xml; do dir=${mets%.mets.xml}; echo $dir; jq '."by-category".TextRegion."pixel-f1".avg' $dir.eynollah-v0.5-eval/*.json $dir.eynollah-206-noautosize-eval/*.json; done > Korrigierte_Layout_GT/eval-gt2-onlyfg-pixel-f1-v0.5-206-noautosize-$variant.log


done



python Korrigierte_Layout_GT/plot.py | sort -V

