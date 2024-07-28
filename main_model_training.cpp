#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

#include <iomanip>
#include <iostream>

#include <random>
#include <cassert>

#include "global.h"
#include "utilities.h"

#include "DataStructures/VectorClass.h"
#include "DataStructures/Dataset.h"

#include "Models/LinearModel.h"
#include "Models/RnnModel.h"

#include <mpi.h>


NN::type main_aux(int rank, int size,
    unsigned output_neurons, unsigned hidden_neurons,
    NN::type learning_rate, unsigned batch_size,
    std::vector<unsigned> which_lags, std::string foldername_outfiles,
    unsigned shuffling_seed, unsigned dl1_seed = 100, unsigned dl2_seed = 200, NN::type tail_decay = 1.0);


int main(int argc, char* argv[]){

    int rank = 0;
    int size = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout <<  "Process " << rank <<  " out of " << size << std::endl;

    const unsigned SS = 10;  // shuffling_seed

    main_aux(rank, size, 2, 10,  0.0005, 32, {1,2,24}, "MPI_Gigat/OutFiles1",  SS, 100,  200,  1.00);
    main_aux(rank, size, 2, 10,  0.0005, 32, {1,2,24}, "MPI_Gigat/OutFiles2",  SS, 200,  300,  1.00);
    main_aux(rank, size, 2, 10,  0.0005, 32, {1,2,24}, "MPI_Gigat/OutFiles3",  SS, 300,  400,  1.00);
    main_aux(rank, size, 2, 10,  0.0005, 32, {1,2,24}, "MPI_Gigat/OutFiles4",  SS, 400,  500,  1.00);
    main_aux(rank, size, 2, 10,  0.0005, 32, {1,2,24}, "MPI_Gigat/OutFiles5",  SS, 500,  600,  1.00);
    main_aux(rank, size, 2, 10,  0.0005, 32, {1,2,24}, "MPI_Gigat/OutFiles6",  SS, 600,  700,  1.00);
    main_aux(rank, size, 2, 10,  0.0005, 32, {1,2,24}, "MPI_Gigat/OutFiles7",  SS, 700,  800,  1.00);
    main_aux(rank, size, 2, 10,  0.0005, 32, {1,2,24}, "MPI_Gigat/OutFiles8",  SS, 800,  900,  1.00);
    main_aux(rank, size, 2, 10,  0.0005, 32, {1,2,24}, "MPI_Gigat/OutFiles9",  SS, 900,  1000, 1.00);
    main_aux(rank, size, 2, 10,  0.0005, 32, {1,2,24}, "MPI_Gigat/OutFiles10", SS, 1000, 1100, 1.00);

    MPI_Finalize();

    return 0;

}


NN::type main_aux(int rank, int size,
  unsigned output_neurons, unsigned hidden_neurons,
  NN::type learning_rate, unsigned batch_size,
  std::vector<unsigned> which_lags, std::string foldername_outfiles,
  unsigned shuffling_seed, unsigned dl1_seed, unsigned dl2_seed, NN::type tail_decay){

    /**************************************************************************************************/
    /*************************************   DESIGNING the MODEL   ************************************/
    /**************************************************************************************************/

    /*
     * The design phase is structured as follows:
     *  1) Initial Settings:
     *  2) Output Format Settings:
     *  3) Data Importation:
     */

    /******************************************** INITIAL SETTINGS *******************************************/

    // Define name of the dataset
    std::string filename_ = "Data/final_dataset.csv";

    // Define directory name for output files (make sure it exists! the program does not check it...)
    std::string foldername_ = foldername_outfiles;

    // Set the number of decimal digits to be shown (Terminal). It is a global variable.
    NN::TRAINING_LOG_DIGITS = 10;

    // Set the number of decimal digits while printing fitted values/predictions. It is a global variable.
    NN::PREDICTION_DIGITS = 8;

    /*************************************** IMPORTING DATA *********************************************/

    // Import training and test set, keeping just relevant rows
    //    -> datetime format: "YYYY-MM-DD hh:mm:ss"
    // Here, we ignore returned indexes (i.e. the [numeric] position of the elements
    // in the dataframe that were kept), since we are not interested in them now
    Dataset dt_train(filename_);
    std::ignore = dt_train.reduce_dataset(">=", "2008-01-01 00:00:00");
    std::ignore = dt_train.reduce_dataset("<",  "2012-01-01 00:00:00");

    Dataset dt_test(filename_);
    std::ignore = dt_test.reduce_dataset(">=", "2011-12-30 00:00:00");
    std::ignore = dt_test.reduce_dataset("<",  "2013-01-01 00:00:00");

    /*******************************  REGRESSORS and REGRESSAND  ***********************************/

    // Define column of exogenous inputs
    const std::string REGRESSAND_VARIABLE = "Demand";

    // Define columns of exogenous inputs for GLM
    const std::vector<std::string> GLM_INPUT_COLUMNS = {
            "Intercept",
            "SY1", "CY1", "SY2", "CY2",
            "Trend",
            "DoW_5","DoW_6", "Holiday"   //(Sat,Sun,Hol)
    };

    // Define columns of exogenous inputs for NN
    const std::vector<std::string> NN_INPUT_COLUMNS = {
            "Drybulb", "Dewpnt",

            "DoW_0", "DoW_1", "DoW_2", "DoW_3","DoW_4",      //(Mon,...,Fri)
            "DoW_5", "DoW_6",  "Holiday",                    //(Sat,Sun,Hol)

            "SY1", "CY1", "SY2", "CY2",
            "SD1", "CD1", "SD2", "CD2"
    };

    /***********************************  RNN SETTINGS  ************************************/

    // Network Hyper-parameters

    // set whether to use Point (1 output node) or Probabilistic (2 output nodes) Forecasting
    NN::OUTPUT_NODES = output_neurons;

    // Network Topology
    const unsigned HIDDEN_NEURONS = hidden_neurons;           // neurons in the first hidden layer
    const std::vector<unsigned> WHICH_LAGS = which_lags;      // which lags to use as feedback

    // Training and optimization
    const NN::type LEARNING_RATE = learning_rate;           // learning rate
    const unsigned NUM_EPOCHS = 1000;                       // maximum number of epochs for training
    const unsigned BATCH_SIZE = batch_size;                 // number of sub-sequences in each mini-batch
    const NN::type LR_DECAY = 0.0;                          // decay of learning rate

    // Stopping criterion
    const unsigned PATIENCE = 100;
    const NN::type MIN_DELTA = 0;
    const unsigned MIN_EPOCHS = 250;
    const bool RESTORE_BEST = true;

    // Windowing of Data
    const unsigned SEQUENCE_LENGTH = 49;        // length of each sequence (fed into the RNN)
    const unsigned WINDOW_SHIFT = 1;            // distance between the beginning of two consecutive sequences

    // Seeds for first and second layer and shuffling random engine
    const unsigned DL1_SEED = dl1_seed;
    const unsigned DL2_SEED = dl2_seed;
    const unsigned SHUFFLING_SEED = shuffling_seed;

    // Regularization. NB: regularization is actually implemented as WEIGHT DECAY
    // cf. "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
    // and references therein (in particular: Hanson & Pratt, 1988)
    const NN::type L1_REGULARIZATION = 0.0;                 // L1 regularization
    const NN::type L2_REGULARIZATION = 0.0;                 // L2 regularization


    /**************************************************************************************************/
    /*******************************************   CHECKS   *******************************************/
    /**************************************************************************************************/

    assert("Unsupported output size" &&
        (NN::OUTPUT_NODES == 1 || NN::OUTPUT_NODES == 2 ) );

    /**************************************************************************************************/
    /************************************ SETTING OUTPUT FORMAT ***************************************/
    /**************************************************************************************************/

    // set the number of digits to be shown.
    std::cout << std::setprecision(NN::TRAINING_LOG_DIGITS) << std::fixed;

    // open the output stream and set the number precision
    std::ofstream output_log(foldername_ + "/training_stats.csv");
    output_log << std::setprecision(NN::TRAINING_LOG_DIGITS) << std::fixed;


    /**************************************************************************************************/
    /*************************************   TRAINING the MODEL   *************************************/
    /**************************************************************************************************/
    /*
     * The training phase is composed by the following operations:
     * 1) Importation of Datasets and normalization of the data
     * 2) Creation of a General Linear Model
     */


    /************************ IMPORTING DATASETS and NORMALIZING  ***************************/

    // store the (observed!) target values for convenience
    std::vector<NN::type> target_train = dt_train.extract_vector(REGRESSAND_VARIABLE);
    std::vector<NN::type> target_test = dt_test.extract_vector(REGRESSAND_VARIABLE);

    // store the corresponding dates
    std::vector<std::string> dates_train = dt_train.get_datetime();
    std::vector<std::string> dates_test = dt_test.get_datetime();

    // apply logarithmic transformation to target variable (consumption)
    dt_train.apply_log(REGRESSAND_VARIABLE);
    dt_test.apply_log(REGRESSAND_VARIABLE);

    // minmax normalize the dataframes (avoid normalizing "Hour" and "Intercept")
    const std::vector<std::string> EXCEPT_COLUMNS = {"Hour", "Intercept"};
    dt_train.normalize_dataframe( EXCEPT_COLUMNS );
    dt_test.normalize_dataframe( dt_train.get_minmax() );


    /***************************** GENERAL LINEAR MODEL  ***********************************/

    // creating vectors for collecting the Linear part of the hybrid model
    std::vector<NN::type> GLM_fitted(dt_train.get_nrows());
    std::vector<NN::type> GLM_predicted(dt_test.get_nrows());

    // looping over the 24 hours (for removing macroscopic seasonality)
    for (unsigned hh = 0; hh < 24; ++hh){

        // create copies of the datasets that will be modified
        Dataset dt_train_GLM = dt_train;
        Dataset dt_test_GLM = dt_test;

        // reducing the new datasets (and store the survived indexes)
        auto glm_idx_train = dt_train_GLM.reduce_dataset("Hour", "==", hh);
        auto glm_idx_test  = dt_test_GLM.reduce_dataset("Hour", "==", hh);

        // extracting features
        auto x_train_GLM = dt_train_GLM.extract_matrix(GLM_INPUT_COLUMNS);
        auto x_test_GLM = dt_test_GLM.extract_matrix(GLM_INPUT_COLUMNS);

        // extracting features
        std::vector<NN::type> y_train_GLM = dt_train_GLM.extract_vector(REGRESSAND_VARIABLE);
        std::vector<NN::type> y_test_GLM = dt_test_GLM.extract_vector(REGRESSAND_VARIABLE);

        // set flag for outliers removal (using IQR)
        const bool REMOVE_OUTLIERS = true;

        // creating and fitting the LinearModel object
        LinearModel lm;
        lm.fit(x_train_GLM, y_train_GLM, REMOVE_OUTLIERS);

        //extracting predictions
        std::vector<NN::type> tmp_fitted = lm.predict(x_train_GLM);
        std::vector<NN::type> tmp_predicted = lm.predict(x_test_GLM);

        //inserting the forecasts in the full vectors, using stored indexes
        for (unsigned i = 0; i < glm_idx_train.size(); ++i)
            GLM_fitted[ glm_idx_train[i] ] = tmp_fitted[i];
        for (unsigned i = 0; i < glm_idx_test.size(); ++i)
            GLM_predicted[ glm_idx_test[i] ] = tmp_predicted[i];

    }


    /**********************************  CREATE RNN  ************************************/

    // the size of NN exogenous inputs is stored
    const unsigned INPUT_NEURONS = NN_INPUT_COLUMNS.size();

    // Initializing Neural Network
    RnnModel rnn(INPUT_NEURONS, HIDDEN_NEURONS, NN::OUTPUT_NODES,
                 WHICH_LAGS, DL1_SEED, DL2_SEED);

    // set activation function for hidden states
    rnn.set_activation("Sigmoid");

    // select metric and loss
    if (NN::OUTPUT_NODES == 1){
        rnn.set_loss("MSE");
        rnn.set_metric("None");
    }
    else{ // OUTPUT_NEURONS == 2

       rnn.set_loss("LogLikelihood");
       //rnn.set_loss("GaussianCRPSLambda", tail_decay);
       //rnn.set_loss("GaussianWinkler", tail_decay);

        //rnn.set_metric("MSE_metric");
        rnn.set_metric("None");
    }

    // set algorithm for computation of gradients
    rnn.set_gradientalgo("TRRL", SEQUENCE_LENGTH);

    // set optimizer and related hyperparameters
    rnn.set_optimizer("Adam", LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, LR_DECAY);

    // add a stopping criterion
    rnn.add_stopping_criterion(PATIENCE, MIN_DELTA, MIN_EPOCHS, RESTORE_BEST);

    // rnn parallelize
    rnn.mpi_parallelize(rank, size);

    /************************* EXTRACTING RNN INPUTS and TARGETS  *******************************/

    // selected columns are extracted and stored
    std::vector<VectorClass> x_train_NN = dt_train.extract_matrix(NN_INPUT_COLUMNS);
    std::vector<VectorClass> x_test_NN = dt_test.extract_matrix(NN_INPUT_COLUMNS);

    // extract regressand variable
    std::vector<NN::type> residual_train = dt_train.extract_vector(REGRESSAND_VARIABLE);
    std::vector<NN::type> residual_test = dt_test.extract_vector(REGRESSAND_VARIABLE);

    // and remove the GLM part
    for (unsigned i = 0; i < residual_train.size(); ++i)
        residual_train[i] -= GLM_fitted[i];
    for (unsigned i = 0; i < residual_test.size(); ++i)
        residual_test[i] -= GLM_predicted[i];

    // print some relevant info
    if (rank == 0){
        std::cout << "\n" << "Training set [fitted values]:" << "\n"
                  << "- first datetime: " << dt_train.get_datetime().at(SEQUENCE_LENGTH-1)  << "\n"
                  << "- last datetime: "  << dt_train.get_datetime().back() << "\n"
                  << "- number of elements: " << dt_train.get_nrows() << std::endl;

        std::cout << "Test set [predicted values]:" << "\n"
                  << "- first datetime: " << dt_test.get_datetime().at(SEQUENCE_LENGTH-1)  << "\n"
                  << "- last datetime: "  << dt_test.get_datetime().back() << "\n"
                  << "- number of elements: " << dt_test.get_nrows() << "\n" << std::endl;
    }

    /************************ CREATING SEQUENCES for FEEDING RNN  ******************************/

    // initialize structures
    std::vector< std::vector<VectorClass> > x_sequences_train, x_sequences_test;
    std::vector<NN::type> y_sequences_train, y_sequences_test;

    // creating the subsequences (of length SEQUENCE_LENGTH and with step WINDOW_SHIFT)
    split_dataset(x_train_NN, residual_train, x_sequences_train, y_sequences_train, SEQUENCE_LENGTH, WINDOW_SHIFT);
    split_dataset(x_test_NN,  residual_test,  x_sequences_test,  y_sequences_test,  SEQUENCE_LENGTH, WINDOW_SHIFT);

    // tic: start measuring training time
    auto t1 = std::chrono::high_resolution_clock::now();

    // training the RNN
    rnn.fit(x_sequences_train, y_sequences_train, output_log, SHUFFLING_SEED);
    
    // toc: print the elapsed time for the NN training
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    // print elapsed time
    output_log << "\n" << "Elapsed time [RNN training]: " << duration << " seconds" << std::endl;


    /******************************************* EXPORT PREDICTION ****************************************/

    // initialize where to store backtesting 95%
    NN::type backtest95 = 0.0;

    if (rank==0) {

        /******************************** INITIALIZING and ADJUSTING **************************************/

        // fitted and predicted
        auto fitted_RNN = rnn.predict(x_sequences_train);
        auto predicted_RNN = rnn.predict(x_sequences_test);

        // declaration
        std::vector<NN::type> mean_fitted(fitted_RNN.size());
        std::vector<NN::type> stdev_fitted(fitted_RNN.size());
        std::vector<NN::type> point_fitted(fitted_RNN.size());

        std::vector<NN::type> mean_predicted(predicted_RNN.size());
        std::vector<NN::type> stdev_predicted(predicted_RNN.size());
        std::vector<NN::type> point_predicted(predicted_RNN.size());

        // necessary for de-normalization
        const auto MINMAX = dt_train.get_minmax(REGRESSAND_VARIABLE);

        // aligning vectors (discarding first "SEQUENCE_LENGTH-1" elements)
        GLM_fitted.erase(GLM_fitted.begin(), GLM_fitted.begin() + SEQUENCE_LENGTH - 1);
        GLM_predicted.erase(GLM_predicted.begin(), GLM_predicted.begin() + SEQUENCE_LENGTH - 1);

        target_train.erase(target_train.begin(), target_train.begin() + SEQUENCE_LENGTH - 1);
        target_test.erase(target_test.begin(), target_test.begin() + SEQUENCE_LENGTH - 1);

        dates_train.erase(dates_train.begin(), dates_train.begin() + SEQUENCE_LENGTH - 1);
        dates_test.erase(dates_test.begin(), dates_test.begin() + SEQUENCE_LENGTH - 1);


        /********************************************* PRINTING *******************************************/

        // open output streams
        std::ofstream outstreamFitted(foldername_ + "/fitted.csv");
        outstreamFitted << std::setprecision(NN::PREDICTION_DIGITS) << std::fixed;
        outstreamFitted << "Date," << REGRESSAND_VARIABLE <<
            ",Forecast,GLM,Mu,Sigma" << std::endl;

        std::ofstream outstreamPredicted(foldername_ + "/predicted.csv");
        outstreamPredicted << std::setprecision(NN::PREDICTION_DIGITS) << std::fixed;
        outstreamPredicted << "Date," << REGRESSAND_VARIABLE <<
            ",Forecast,GLM,Mu,Sigma" << std::endl;

        // process training set
        for (unsigned seq = 0; seq < fitted_RNN.size(); ++seq) {

            // export predicted values
            NN::type glm = GLM_fitted[seq];
            NN::type mu = fitted_RNN[seq](0);
            NN::type sigma = (NN::OUTPUT_NODES==2) ? fabs(fitted_RNN[seq](1)) : 0.0;

            // denormalizing (minmax)
            glm = glm * (MINMAX.second - MINMAX.first) + MINMAX.first;
            mu  = mu  * (MINMAX.second - MINMAX.first);
            sigma = sigma * (MINMAX.second - MINMAX.first);

            mean_fitted[seq] = glm + mu;
            stdev_fitted[seq] = sigma;
            point_fitted[seq] = exp(mean_fitted[seq] + 0.5 * stdev_fitted[seq] * stdev_fitted[seq]);

            outstreamFitted << dates_train[seq] << ","
                            << target_train[seq] << ","
                            << point_fitted[seq] << ","
                            << glm << ","
                            << mean_fitted[seq] << ","
                            << stdev_fitted[seq] << std::endl;
        }

        // process testing set
        for (unsigned seq = 0; seq < predicted_RNN.size(); ++seq) {

            // export predicted values
            NN::type glm = GLM_predicted[seq];
            NN::type mu = predicted_RNN[seq](0);
            NN::type sigma = (NN::OUTPUT_NODES==2) ? fabs(predicted_RNN[seq](1)) : 0.0;

            // denormalizing (minmax)
            glm = glm * (MINMAX.second - MINMAX.first) + MINMAX.first;
            mu  = mu  * (MINMAX.second - MINMAX.first);
            sigma = sigma * (MINMAX.second - MINMAX.first);

            mean_predicted[seq] = glm + mu;
            stdev_predicted[seq] = sigma;
            point_predicted[seq] = exp(mean_predicted[seq] + 0.5 * stdev_predicted[seq] * stdev_predicted[seq]);

            outstreamPredicted  << dates_test[seq]  << ","
                                << target_test[seq] << ","
                                << point_predicted[seq] << ","
                                << glm << ","
                                << mean_predicted[seq] << ","
                                << stdev_predicted[seq] << std::endl;

        }

        /********************************************* STATISTICS *******************************************/

        output_log << "\n" << "Tail Decay: " << tail_decay << std::endl;

        output_log << "\n" << "Training Set:" << std::endl;
        compute_statistics(mean_fitted, stdev_fitted, point_fitted,
                           target_train, output_log);

        output_log << "\n" << "Test Set:" << std::endl;
        backtest95 = compute_statistics(mean_predicted, stdev_predicted, point_predicted,
                           target_test, output_log);

        for (int receiver = 1; receiver < size; ++receiver)
            MPI_Send(&backtest95, 1, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);

    }
    else{ // i.e., if (rank!=0)
        MPI_Recv(&backtest95, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return backtest95;

}
