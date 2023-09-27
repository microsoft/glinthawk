rm $4
for slices in 1 4
do
	for batch in {1..32}
	do

		value=$($1 $2 $3 1 $slices $batch 2>&1 | grep "Token Generation" | grep -o 'avg=[[:space:]]*[0-9.]*' | awk -F= '{print $2}')
		if [ -n "$value" ]; then
    			echo "Slices: $slices and $batch Batch size: $value" >> "$4"
		else
    			echo "Line not found" >> "$4"
		fi
	done
done
for slices in 31
do
        for batch in {1..32}
        do

                value=$($1 $2 $3 $slices 31 $batch 2>&1 | grep "Token Generation" | grep -o 'avg=[[:space:]]*[0-9.]*' | awk -F= '{print $2}')
                if [ -n "$value" ]; then
                        echo "Slices: $slices and $batch Batch size: $value" >> "$4"
                else
                        echo "Line not found" >> "$4"
                fi
        done
done
