# RNN_v8_muac_weather
RNN models to predict W-Weather regulation over the MUAC region

## Comments
The following elements/files/folders are mandatory. However, for confidentiality they must be privat: 

1. The regulations you want to study must be in a `CSV` file;

2. Inside the repository must be the folder `Exports_RNEST` which contains the files exported
    from RNEST which contain the raw scalar values;
   
3. Inside the repository must be the folder `Export_weather_information` which contains the files exported
    from weather API (ERA 5) which contain the raw weather scalar values;
   
        To extract the raw weather scalar variables you should use:
            Extract_weather_features_per_sector_per_monthAndDays.ipynb
   
3. When pre-processed the information and created the input samples, they must be saved in the
    folder `Counting_variables`. This speed up the different proofs of concept over a given TV.


