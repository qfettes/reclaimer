cd ../
python3 master_data_collect_ath_hotel.py --user-name bbeckwi2 \
	--stack-name hotelreservation \
	--min-users 5 --max-users 400 --users-step 5 \
	--exp-time 750 --measure-interval 1 --slave-port 40011 --deploy-config hotel_cluster.json \
	--mab-config hotel_mab.json --deploy
