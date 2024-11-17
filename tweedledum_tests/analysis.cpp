#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h> // rand()
#include <time.h> // time()
#include <chrono> // time()

#include <mockturtle/io/dimacs_reader.hpp>
#include <mockturtle/io/write_dot.hpp>
#include <mockturtle/networks/xag.hpp>
#include <tweedledum/Synthesis/xag_synth.h>
#include <tweedledum/Synthesis/lhrs_synth.h>
#include <tweedledum/Utils/Visualization/string_utf8.h>

using namespace std;

int rbool(){
/* generate either 0 or 1 with equal probability */
    return rand() % 2;
}

int rint(int i){
/* generate random int between 1 and i, including i */
    return rand() % i + 1;
}

struct Timer{
    Timer(){
        start_time = chrono::high_resolution_clock::now();
    }

    double get_duration(){
        auto end_time = chrono::high_resolution_clock::now();

        auto start = chrono::time_point_cast<chrono::microseconds>(start_time).time_since_epoch().count();
        auto end = chrono::time_point_cast<chrono::microseconds>(end_time).time_since_epoch().count();
        auto duration = end - start;
        double ms = duration * 0.001;
        return ms;
    }

private:
    chrono::time_point<chrono::high_resolution_clock> start_time;
};

string get_timestamp(){
    /* get string with timestamp*/
    time_t rawtime;
    struct tm * timeinfo;
    char buffer [80];
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    strftime(buffer,80,"%m-%d_%H:%M:%S",timeinfo);
    string timestamp = buffer;
    return timestamp;
}

struct TableFile{
    ofstream file;
    int file_counter = 0;
    string base_filename;
    vector<string> col_names;
    int n_cols;
    int n_lines;
    vector<int> col_widths;

    TableFile(string folder, vector<tuple<string, int>> _columns){
        base_filename = folder + "/" + get_timestamp() + "_";

        for(auto& col : _columns){
            col_names.push_back(get<0>(col));
            col_widths.push_back(get<1>(col));
        }
        n_cols = col_names.size();
    }

    void open(){
        n_lines = 0;
        string s = to_string(file_counter);
        string filename = base_filename + string(10 - s.length(), '0') + s + ".dat";
        file.open(filename, ios::app);

        for(int i=0; i<n_cols; i++){
            file << setw(col_widths[i]) << col_names[i];
        }
        file << "\n";
    }

    void close(){
        file.close();
        file_counter++;
    }


    void write_row(vector<int*> data_point){

        if(!file.is_open())
            open();

        for(int i=0; i<n_cols; i++){
            file << setw(col_widths[i]) << *data_point[i];
        }
        file << "\n";
        file.flush();

        n_lines++;

        if(n_lines > 100){
            close();
        }
    }
};

string random_SAT(const int N, int K, const int M){
    /* returns a string with a random CNF-SAT instance of N variables,
     * M clauses and K literals per clause, in DIMACS format */

    string dimacs ("p cnf " + to_string(N) + " " + to_string(M));

    /* for each clause*/
    for(int m = 0; m<M; m++){

        /* choose K DISTINCT variables */ 
        vector<int> clause_vars (K, 0);
        clause_vars[0] = rint(N);

        for(int k=1; k<K;){
            int new_var = rint(N);

            /* if the element is not in the vector clause_vars, add it and increment k*/
            if (find(begin(clause_vars), end(clause_vars), new_var) == end(clause_vars)){
                clause_vars[k] = new_var;
                k++;
            }
        }

        /* create clause string and append it to dimacs string */
        string clause ("\n");
        for(int k=0; k<K; k++){
            clause += (rbool() ? "" : "-") + to_string(clause_vars[k]) + " ";
        }
        clause += "0";

        dimacs += clause;
    }
    return dimacs;
}

template<class Ntk, class Drawer = mockturtle::default_dot_drawer<Ntk>>
int ntk_num_edges(Ntk const& ntk, Drawer const& drawer = {}){
    /* return number of edges in a network */
    int e = 1;
    ntk.foreach_node( [&]( auto const& n ) {
        if ( !ntk.is_constant( n ) && !ntk.is_pi( n ) ){
            ntk.foreach_fanin( n, [&]( auto const& f ) {
                if ( !drawer.draw_signal( ntk, n, f ) )
                    return true;
                e++;
                return true;
            });
        }
    });
    return e;
}

struct Random_sat_xag{
    int N;
    int K;
    int M;
    int num_nodes;
    int num_edges;

    Random_sat_xag(const int& n, const int& k, const int& m) : N(n), K(k), M(m) {
        stringstream ss(random_SAT(N, K, M));
        mockturtle::xag_network xag;
        auto r = lorina::read_dimacs(ss, mockturtle::dimacs_reader(xag));
        num_nodes = xag.size();
        num_edges = ntk_num_edges(xag);
    }
};

struct Mean{
    int value = 0;
    int N = 0;
    
    Mean(){}

    void sample(int& sample){
        N++;
        value += (sample - value) / (N+1);
    }
};

struct Max{
    int value = 0;

    Max(){}

    void sample(int& sample){
        if(sample > value)
            value = sample;
    }
};

void sim1(){
    /* In this simulation I determine the average and maximum number of nodes
     * and number of edges, for (N,K,M) instances, using n_samples for each data point */

    vector<tuple<string, int>> columns{
        {"N", 10}, 
        {"K", 10},
        {"M", 10},
        {"mean_num_nodes", 17},
        {"max_num_nodes", 17},
        {"mean_num_edges", 17},
        {"max_num_edges", 17}
    };

    TableFile tf("/home/hugens/shared/uni/project/tweedledum/examples/data", columns);

    int n_samples = 10000;

    for(int K=3; K<15; K++){
        int N_min = 150;
        int N_max = 500;
        //int N_step = (N_max - N_min) / 30;
        int N_step = 20;

        for(int N=N_min; N<N_max; N+=N_step){
            int M_min = 2*N;
            int M_max = 8*N;
            int M_step = (M_max - M_min) / 50;

            for(int M=M_min; M<M_max; M+=M_step){

                cout << "N=" << N 
                     << " K=" << K 
                     << " M=" << M 
                     << " samples=" << n_samples
                     << "\n";
                Mean mean_nodes;
                Mean mean_edges;
                Max max_nodes;
                Max max_edges;

                for(int i=0; i<n_samples; i++){
                    Random_sat_xag sat_xag(N, K, M); 
                    mean_nodes.sample(sat_xag.num_nodes);
                    mean_edges.sample(sat_xag.num_edges);
                    max_nodes.sample(sat_xag.num_nodes);
                    max_edges.sample(sat_xag.num_edges);
                }

                vector<int*> data_point {
                    &N, 
                    &K, 
                    &M, 
                    &mean_nodes.value, 
                    &max_nodes.value, 
                    &mean_edges.value, 
                    &max_edges.value, 
                };

                tf.write_row(data_point);
            }
        }
    }
    tf.close();
}

void sim2(int N, int K, int M){
    /* In this simulation, I study how the (estimated) probabililty distribution of
     * number of nodes and edges for random K-SAT changes as we increase n_samples,
     * the ideia being that maybe using n_samples=10000 in the previous simulation
     * might be overkill, and if so what is an adequate n_samples? */

    /* 2**20 = 1048576 */
    int max_n_samples = pow(2, 19); 
    int flushing_n_samples = pow(2, 4);
    int main_loop_max = max_n_samples/flushing_n_samples;

    ofstream file;
    string filename = "/home/hugens/shared/uni/project/tweedledum/examples/data/" 
                      + get_timestamp() 
                      + "_N=" + to_string(N)
                      + "_K=" + to_string(K)
                      + "_M=" + to_string(M)
                      + ".dat";

    file.open(filename, ios::app);
    file << "num_nodes" << "," << "num_edges" << "\n";

    for(int n_samples=0; n_samples < main_loop_max; n_samples++){
        for(int i=0; i < flushing_n_samples; i++){
            Random_sat_xag sat_xag(N, K, M); 
            file << sat_xag.num_nodes << "," << sat_xag.num_edges << "\n";
        }
        file.flush();
        cout << "\r [" << setw(3) << round((double)n_samples * 100 /main_loop_max)  << "%]" << flush;
    }

    file.close();
}

int main(){
    /* seed random number generator */
    srand (time(NULL));

    sim2(20, 3, 85);
    sim2(30, 3, 85);
    sim2(20, 5, 85);
    sim2(20, 3, 120);
    sim2(300, 3, 1290);
}
