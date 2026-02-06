This builds a docker image, currently no target docker repo.

To build run ./build_docker_image

To do a forward prediction you need a file with PREFIX_train.csv and PREFIX_future.csv.

The future file contains datetime followed by predicted or known exog data columns.

The train file has datetime, price followed by exog data columns.

To run the container you need to mount the file under /usr/src/app/data and provide the prefix for the supplied CSV files:

docker run --rm --volume=../data:/usr/src/app/data -e DATA_NAME=all_exog price_predict:latest

The output is to stout. If you require a file to be generated you can provide env OP_FILE which will be written in the data directory:

docker run --rm --volume=../data:/usr/src/app/data -e DATA_NAME=all_exog -e OP_FILE=banana.csv price_predict:latest