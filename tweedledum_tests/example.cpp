#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <vector>
#include <algorithm>

#include <mockturtle/io/dimacs_reader.hpp>
#include <mockturtle/io/write_dot.hpp>
#include <mockturtle/networks/xag.hpp>
#include <tweedledum/Synthesis/xag_synth.h>
#include <tweedledum/Synthesis/lhrs_synth.h>
#include <tweedledum/Utils/Visualization/string_utf8.h>

using namespace std;

/* generate either 0 or 1 with equal probability */
int rbool(){
    return rand() % 2;
}

/* generate random int between 1 and i, including i */
int rint(int i){
    return rand() % i + 1;
}

string random_SAT(const int N, int K, const int M){
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
int num_edges(Ntk const& ntk, Drawer const& drawer = {}){
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

struct sat
{
    int N;
    int K;
    int M;
    int xag_num_nodes;
    int xag_num_edges;
    int oracle_num_qubits;
    int oracle_num_instr;

    sat(const int n, const int k, const int m, const int depth = 0) : N(n), K(k), M(m) {
        std::stringstream ss(random_SAT(N, K, M));
        mockturtle::xag_network xag;
        lorina::read_dimacs(ss, mockturtle::dimacs_reader(xag));

        xag_num_nodes = xag.size();
        xag_num_edges = num_edges(xag);

        if(depth == 1){
            tweedledum::Circuit oracle = tweedledum::xag_synth(xag);
            oracle_num_qubits = oracle.num_qubits();
            oracle_num_instr = oracle.num_instructions();
        }
    }
};

void test1(){
    /* create a xag from a dimacs string and print it*/
    mockturtle::xag_network xag;
    std::string filename ("/home/hugens/shared/uni/project/tweedledum/examples/3sat.dimacs");
    lorina::read_dimacs(filename, mockturtle::dimacs_reader(xag));
    mockturtle::write_dot(xag, std::cout);
}

void test2(){
    /* check if this dimacs instance gives same oracle in both python and cpp implementations*/
    std::string dimacs_str ("p cnf 3 5\n-1 -2 -3 0\n1 -2 3 0\n1 2 -3 0\n1 -2 -3 0\n-1 2 3 0");
    std::stringstream ss(dimacs_str);
    mockturtle::xag_network xag;
    lorina::read_dimacs(ss, mockturtle::dimacs_reader(xag));
    tweedledum::Circuit oracle = tweedledum::xag_synth(xag);
    tweedledum::print(oracle, 1000);
}

void test3(){
    /* extracting number of qubits and circuit depth */
    std::string dimacs_str ("p cnf 3 5\n-1 -2 -3 0\n1 -2 3 0\n1 2 -3 0\n1 -2 -3 0\n-1 2 3 0");
    std::stringstream ss(dimacs_str);
    mockturtle::xag_network xag;
    lorina::read_dimacs(ss, mockturtle::dimacs_reader(xag));
    tweedledum::Circuit oracle = tweedledum::xag_synth(xag);
    tweedledum::print(oracle, 1000);

    cout << "num_qubits = " << oracle.num_qubits() << "\n";
    cout << "num_instructions = " << oracle.num_instructions() << "\n";
}

void test4(){
    /* this test shows that sizeof(xag) returns the size of the mockturtle::xag_network type
     * and not necessarily of what is stored in that type */
    std::string dimacs_str ("p cnf 3 5\n-1 -2 -3 0\n1 -2 3 0\n1 2 -3 0\n1 -2 -3 0\n-1 2 3 0");
    std::stringstream ss(dimacs_str);
    mockturtle::xag_network xag;
    lorina::read_dimacs(ss, mockturtle::dimacs_reader(xag));
    cout << sizeof(xag) << endl;

    std::stringstream ss1(random_SAT(300, 3, 1290));
    mockturtle::xag_network xag1;
    lorina::read_dimacs(ss1, mockturtle::dimacs_reader(xag1));
    cout << sizeof(xag1) << endl;
}

void test5(){
    /* since memory usage of a XAG seems to be impossible to 
     * figure out, let's focus on network size instead */
    std::string dimacs_str ("p cnf 3 5\n-1 -2 -3 0\n1 -2 3 0\n1 2 -3 0\n1 -2 -3 0\n-1 2 3 0");
    std::stringstream ss(dimacs_str);
    mockturtle::xag_network xag;
    lorina::read_dimacs(ss, mockturtle::dimacs_reader(xag));
    mockturtle::write_dot(xag, std::cout);
    cout << "size = " << xag.size() << endl;
    cout << "num_cis = " << xag.num_cis() << endl;
    cout << "num_cos = " << xag.num_cos() << endl;
    cout << "num_latches = " << xag.num_latches() << endl;
    cout << "num_pis = " << xag.num_pis() << endl;
    cout << "num_pos = " << xag.num_pos() << endl;
    cout << "num_registers = " << xag.num_registers() << endl;
    cout << "num_gates = " << xag.num_gates() << endl;
}

void test6(){
    /* getting network edge number */
    //std::string dimacs_str ("p cnf 3 5\n-1 -2 -3 0\n1 -2 3 0\n1 2 -3 0\n1 -2 -3 0\n-1 2 3 0");
    std::string dimacs_str = random_SAT(300, 3, 1290);
    std::stringstream ss(dimacs_str);
    mockturtle::xag_network xag;
    lorina::read_dimacs(ss, mockturtle::dimacs_reader(xag));
    mockturtle::write_dot(xag, std::cout);
    cout << "num_nodes = " << xag.size() << "\n";
    cout << "num_edges = " << num_edges(xag) << "\n";
}

void test7(){
    /* this test shows that with the regular xag_synth mathod for building an oracle circuit
     * from a xag_network, that the following holds:
     * xag_num_nodes - oracle_num_qubits = 1
     * xag_num_edges - oracle_num_instr = 2*/
    sat p (300, 3, 1290, 1);

    cout << "xag_num_nodes " << p.xag_num_nodes << "\n";
    cout << "xag_num_edges " << p.xag_num_edges << "\n";
    cout << "oracle_num_qubits " << p.oracle_num_qubits << "\n";
    cout << "oracle_num_instr " << p.oracle_num_instr << "\n";

    for(int n=20; n<300; n += 20){
        int m_min = 2*n;
        int m_max = 8*n;
        int m_step = (m_max - m_min) / 100;

        for(int m = m_min; m<m_max; m += m_step){
            sat p (n, 5, m, 1);
            if(((p.xag_num_nodes - p.oracle_num_qubits) != 1) && ((p.xag_num_edges - p.oracle_num_instr) != 2))
                cout << "found exception" << "\n";
        }
    }  
}

void test8(){
    /* does network features only depend on N, K, M, or also in the specific instance?
     * it seems like num_nodes and num_edges are almost the same. For fixes N, K, M, 
     * let's find the highest deviation from the moving mean */

    float mean_nodes = 0;
    float mean_edges = 0;
    int max_dev_nodes = 0;
    int max_dev_edges = 0;

    for(int i=0; i<1000000; i++){
        sat p (20, 8, 80);

        mean_nodes += (p.xag_num_nodes - mean_nodes) / (i+1);
        mean_edges += (p.xag_num_edges - mean_edges) / (i+1);

        int dev_edges = p.xag_num_nodes - mean_nodes;
        int dev_nodes = p.xag_num_edges - mean_edges;

        if(dev_nodes > max_dev_nodes){
            max_dev_nodes = dev_nodes;
            cout << "max_dev_nodes " << max_dev_nodes << "\n";
        }

        if(dev_edges > max_dev_edges){
            max_dev_edges = dev_edges;
            cout << "max_dev_edges " << max_dev_edges << "\n";
        }
    }
}

void test9(){
    /* try k-LUT based method */
    std::string dimacs_str ("p cnf 3 5\n-1 -2 -3 0\n1 -2 3 0\n1 2 -3 0\n1 -2 -3 0\n-1 2 3 0");
    std::stringstream ss(dimacs_str);
    mockturtle::xag_network xag;
    lorina::read_dimacs(ss, mockturtle::dimacs_reader(xag));
    tweedledum::Circuit oracle = tweedledum::lhrs_synth(xag);
    tweedledum::print(oracle, 1000);
    cout << oracle.num_qubits() << "\n";
    cout << oracle.num_instructions() << "\n";
}

int main(){

    /* intialize random seed for the rand() generator */
    srand (time(NULL));

    test9();
}
