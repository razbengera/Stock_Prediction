This project is forecasting stock prices. In the first stage it is required to run the above commands. Then we need to enter the browser and insert a stock simble we want to make a forecast (for example IBM, TSLA).
After downloading the information, you will move to another page on the browser in order to make the forecast.

In order to run this project in docker, preform the following commands (yoe need to change the "C:\...":

cd\
cd\Users\razbe\stock_project

docker build . -t forecast_1 -f ./dockerfile
docker run -v  C:\Users\razbe\stock_project:/app/main -p 8080:8080 -it forecast_1

Then you can access http://localhost:8080/ via browser and use the application.

![alt text](https://github.com/razbengera/Stock_Prediction/blob/cd470a4c0c6a8bf720313e3cd283a1e81309bb22/HomePage.jpg?raw=true)

![alt text](https://github.com/razbengera/Stock_Prediction/blob/c82e9ab8c331a91389bdd5ffb017a69cfb064531/StockData.jpg?raw=true)

![alt text](https://github.com/razbengera/Stock_Prediction/blob/c82e9ab8c331a91389bdd5ffb017a69cfb064531/Prediction.jpg?raw=true)
