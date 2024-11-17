Libraries used in tweedledum:
-----
abc - ABC: System for Sequential Logic Synthesis and Formal Verification
bill - C++ header-only reasoning library
Catch2 test framework for unit-tests, TDD and BDD
CMake for build automation
Eigen template library for linear algebra
{fmt} - A modern formatting library
kitty - truth table library
lorina - C++ parsing library for simple formats used in logic synthesis and formal verification
mockturtle - C++ logic network library
nlohmann/json - JSON for Modern C++
parallel_hashmap - A family of header-only, very fast and memory-friendly hashmap and btree containers.
percy - C++ header-only exact synthesis library
pybind11 - Seamless operability between C++11 and Python
rang - A Minimal, Header only Modern c++ library for terminal goodies

Relevant functions:

---------------------------------------------------------------
/tests/Synthesis/xag_synth.cpp

- function to create XAG network from a quantum circuit

```c++
mockturtle::xag_network to_xag_network(Circuit const& circuit, uint32_t num_i, uint32_t num_o)
```

- tests involve :

- creation of a XAG network named oracle:

```c++
auto oracle = mockturtle::xag_network()
auto a = oracle.create_pi();
auto b = oracle.create_pi();
auto ab = oracle.create_and(a, b);
oracle.create_po(ab);
```

- building a circuit from that XAG network using the xag_synth function:

```c++
Circuit circuit = xag_synth(oracle);

- turning it back to a XAG by using the helper function to_xag_network 

```c++
auto xag = to_xag_network(circuit, oracle.num_pis(), oracle.num_pos())
```

- and checking if they are equivalent.

---------------------------------------------------------------
/include/tweedledum/Synthesis/xag_synth.h

This is the header file included in /tests/Synthesis/xag_synth.cpp that has the xag_synth method.

```c++
namespace tweedledum {

void xag_synth(Circuit& circuit, std::vector<Qubit> const& qubits,
  std::vector<Cbit> const& cbits, mockturtle::xag_network const& xag,
  nlohmann::json const& config = {});

Circuit xag_synth(
  mockturtle::xag_network const& xag, nlohmann::json const& config = {});

} // namespace tweedledum
```

---------------------------------------------------------------
/src/Synthesis/xag/xag_synth.cpp
/src/Synthesis/xag/lhrs_synth.cpp

This are the files that really have the implementations of these methods.
---------------------------------------------------------------
/python/tweedledum/classical/mockturtle.cpp

lorina is used to parse a dimacs file into a mockturtle::xag_newtork 

```c++
module.def("read_dimacs", [](std::string const filename) {
    xag_network xag;
    lorina::read_dimacs(filename, dimacs_reader(xag));
    return xag;
}, "Create a LogicNetwork from a DIMACS file.");
```

---------------------------------------------------------------
/external/mockturtle/mockturtle/io/dimacs_reader.hpp

Location of the dimacs reader used above

---------------------------------------------------------------
/external/lorina/lorina/dimacs.hpp

This file might be useful to create SAT instances without having to write them into a file.

Two methods of interest are:

```c++
[[nodiscard]] inline return_code read_dimacs(const std::string& filename, 
                                            const dimacs_reader& reader, 
                                            diagnostic_engine* diag = nullptr )
```
which uses a file as input, and

```c++
[[nodiscard]] inline return_code read_dimacs(std::istream& in,
                                            const dimacs_reader& reader,
                                            diagnostic_engine* diag = nullptr )
```

which uses a istream as input. The second may be the most useful since it may allow to pass dimacs instances witout creating the files themselves.

---------------------------------------------------------------
/python/tweedledum/synthesis/synthesis.cpp

This file has an nice overview of available synthesis methods. The ones that take a XAG network as input.

```c++
module.def("lhrs_synth",
    py::overload_cast<mockturtle::xag_network const&, nlohmann::json const&>(&lhrs_synth),
    py::arg("xag"), py::arg("config") = nlohmann::json(),
    "LUT-based Hierarchical Synthesis from a XAG representation.");
    
module.def("xag_synth", 
    py::overload_cast<mockturtle::xag_network const&, nlohmann::json const&>(&xag_synth),
    py::arg("xag"), py::arg("config") = nlohmann::json(),
    "Synthesize a quantum circuit from a XAG representation.");
```
---------------------------------------------------------------
/src/Utils/Visualization/string_utf8.cpp

Might be a function in this file which prints a quantum circuit.

```c++
void print(Circuit const& circuit, uint32_t const max_rows)
{
    fmt::print("{}", to_string_utf8(circuit, max_rows));
}
```

---------------------------------------------------------------
This is a check my python and cpp codes are returning the same circuit for a specific oracle.

The python code:

```python
from tweedledum import BoolFunction
from tweedledum.synthesis import xag_synth, lhrs_synth

bool_function = BoolFunction.from_dimacs_file('3sat.dimacs')
xag = bool_function.logic_network()
circuit = xag_synth(xag)
print(circuit)
```
outputs:
```
                                                                          ╭────╮     ╭────╮                                                            
__a15 : ──────────────────────────────────────────────────────────────────┤ rx ├──●──┤ rx ├────────────────────────────────────────────────────────────
                                                                    ╭────╮╰─┬──╯  │  ╰─┬──╯╭────╮                                                      
__a14 : ────────────────────────────────────────────────────────────┤ rx ├──┼─────●────┼───┤ rx ├──────────────────────────────────────────────────────
                                                              ╭────╮╰─┬──╯  │     │    │   ╰─┬──╯            ╭────╮                                    
__a13 : ──────────────────────────────────────────────────────┤ rx ├──●─────┼─────┼────┼─────●───────────────┤ rx ├────────────────────────────────────
                                                        ╭────╮╰─┬──╯  │     │     │    │     │               ╰─┬──╯                        ╭────╮      
__a12 : ────────────────────────────────────────────────┤ rx ├──┼─────◯─────┼─────┼────┼─────◯─────────────────┼───────────────────────────┤ rx ├──────
                                                  ╭────╮╰─┬──╯  │           │     │    │   ╭────╮              │                           ╰─┬──╯      
__a11 : ──────────────────────────────────────────┤ rx ├──┼─────┼───────────◯─────┼────◯───┤ rx ├──────────────┼─────────────────────────────┼─────────
                                            ╭────╮╰─┬──╯  │     │           │     │    │   ╰─┬──╯      ╭────╮  │                             │         
__a10 : ────────────────────────────────────┤ rx ├──┼─────┼─────┼───────────◯─────┼────◯─────┼─────────┤ rx ├──┼─────────────────────────────┼─────────
                                      ╭────╮╰─┬──╯  │     │     │                 │          │         ╰─┬──╯  │   ╭────╮                    │         
 __a9 : ──────────────────────────────┤ rx ├──┼─────┼─────┼─────◯─────────────────┼──────────┼───────────┼─────◯───┤ rx ├────────────────────┼─────────
                                ╭────╮╰─┬──╯  │     │     │     │                 │          │           │     │   ╰─┬──╯      ╭────╮        │         
 __a8 : ────────────────────────┤ rx ├──┼─────┼─────┼─────┼─────◯─────────────────┼──────────┼───────────┼─────◯─────┼─────────┤ rx ├────────┼─────────
                          ╭────╮╰─┬──╯  │     │     │     │                       │          │           │           │         ╰─┬──╯        │   ╭────╮
 __a7 : ──────────────────┤ rx ├──┼─────┼─────┼─────┼─────●───────────────────────┼──────────┼───────────┼───────────┼───────────┼───────────●───┤ rx ├
                    ╭────╮╰─┬──╯  │     │     │     │     │                       │          │   ╭────╮  │           │           │           │   ╰─┬──╯
 __a6 : ────────────┤ rx ├──┼─────┼─────┼─────┼─────●─────┼───────────────────────┼──────────●───┤ rx ├──┼───────────┼───────────┼───────────┼─────┼───
              ╭────╮╰─┬──╯  │     │     │     │     │     │                       │          │   ╰─┬──╯  │           │   ╭────╮  │           │     │   
 __a5 : ──────┤ rx ├──┼─────┼─────┼─────●─────┼─────┼─────┼───────────────────────┼──────────┼─────┼─────┼───────────●───┤ rx ├──┼───────────┼─────┼───
        ╭────╮╰─┬──╯  │     │     │     │     │     │     │                       │          │     │     │           │   ╰─┬──╯  │   ╭────╮  │     │   
 __a4 : ┤ rx ├──┼─────┼─────┼─────●─────┼─────●─────┼─────┼───────────────────────┼──────────┼─────┼─────●───────────┼─────┼─────●───┤ rx ├──┼─────┼───
        ╰─┬──╯  │     │     │     │     │     │     │     │                     ╭─┴─╮        │     │     │           │     │     │   ╰─┬──╯  │     │   
 __q3 : ──┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────────────────────┤ x ├────────┼─────┼─────┼───────────┼─────┼─────┼─────┼─────┼─────┼───
          │     │     │     │     │     │     │     │     │                     ╰───╯        │     │     │           │     │     │     │     │     │   
 __q2 : ──●─────◯─────◯─────●─────┼─────┼─────┼─────┼─────┼──────────────────────────────────┼─────◯─────┼───────────┼─────◯─────┼─────●─────┼─────●───
          │     │     │     │     │     │     │     │     │                                  │     │     │           │     │     │     │     │     │   
 __q1 : ──●─────◯─────●─────◯─────┼─────┼─────┼─────┼─────┼──────────────────────────────────┼─────●─────┼───────────┼─────◯─────┼─────●─────┼─────◯───
                                  │     │     │     │     │                                  │           │           │           │           │         
 __q0 : ──────────────────────────◯─────●─────●─────◯─────◯──────────────────────────────────◯───────────●───────────●───────────◯───────────◯─────────
```
and the cpp code
```c++
std::string dimacs_str ("p cnf 3 5\n-1 -2 -3 0\n1 -2 3 0\n1 2 -3 0\n1 -2 -3 0\n-1 2 3 0");
std::stringstream ss(dimacs_str);
mockturtle::xag_network xag;
lorina::read_dimacs(ss, mockturtle::dimacs_reader(xag));
tweedledum::Circuit oracle = tweedledum::xag_synth(xag);
tweedledum::print(oracle, 1000);
```
outputs:
```
                                                                          ╭────╮     ╭────╮                                                            
__a15 : ──────────────────────────────────────────────────────────────────┤ rx ├──●──┤ rx ├────────────────────────────────────────────────────────────
                                                                    ╭────╮╰─┬──╯  │  ╰─┬──╯╭────╮                                                      
__a14 : ────────────────────────────────────────────────────────────┤ rx ├──┼─────●────┼───┤ rx ├──────────────────────────────────────────────────────
                                                              ╭────╮╰─┬──╯  │     │    │   ╰─┬──╯            ╭────╮                                    
__a13 : ──────────────────────────────────────────────────────┤ rx ├──●─────┼─────┼────┼─────●───────────────┤ rx ├────────────────────────────────────
                                                        ╭────╮╰─┬──╯  │     │     │    │     │               ╰─┬──╯                        ╭────╮      
__a12 : ────────────────────────────────────────────────┤ rx ├──┼─────◯─────┼─────┼────┼─────◯─────────────────┼───────────────────────────┤ rx ├──────
                                                  ╭────╮╰─┬──╯  │           │     │    │   ╭────╮              │                           ╰─┬──╯      
__a11 : ──────────────────────────────────────────┤ rx ├──┼─────┼───────────◯─────┼────◯───┤ rx ├──────────────┼─────────────────────────────┼─────────
                                            ╭────╮╰─┬──╯  │     │           │     │    │   ╰─┬──╯      ╭────╮  │                             │         
__a10 : ────────────────────────────────────┤ rx ├──┼─────┼─────┼───────────◯─────┼────◯─────┼─────────┤ rx ├──┼─────────────────────────────┼─────────
                                      ╭────╮╰─┬──╯  │     │     │                 │          │         ╰─┬──╯  │   ╭────╮                    │         
 __a9 : ──────────────────────────────┤ rx ├──┼─────┼─────┼─────◯─────────────────┼──────────┼───────────┼─────◯───┤ rx ├────────────────────┼─────────
                                ╭────╮╰─┬──╯  │     │     │     │                 │          │           │     │   ╰─┬──╯      ╭────╮        │         
 __a8 : ────────────────────────┤ rx ├──┼─────┼─────┼─────┼─────◯─────────────────┼──────────┼───────────┼─────◯─────┼─────────┤ rx ├────────┼─────────
                          ╭────╮╰─┬──╯  │     │     │     │                       │          │           │           │         ╰─┬──╯        │   ╭────╮
 __a7 : ──────────────────┤ rx ├──┼─────┼─────┼─────┼─────●───────────────────────┼──────────┼───────────┼───────────┼───────────┼───────────●───┤ rx ├
                    ╭────╮╰─┬──╯  │     │     │     │     │                       │          │   ╭────╮  │           │           │           │   ╰─┬──╯
 __a6 : ────────────┤ rx ├──┼─────┼─────┼─────┼─────●─────┼───────────────────────┼──────────●───┤ rx ├──┼───────────┼───────────┼───────────┼─────┼───
              ╭────╮╰─┬──╯  │     │     │     │     │     │                       │          │   ╰─┬──╯  │           │   ╭────╮  │           │     │   
 __a5 : ──────┤ rx ├──┼─────┼─────┼─────●─────┼─────┼─────┼───────────────────────┼──────────┼─────┼─────┼───────────●───┤ rx ├──┼───────────┼─────┼───
        ╭────╮╰─┬──╯  │     │     │     │     │     │     │                       │          │     │     │           │   ╰─┬──╯  │   ╭────╮  │     │   
 __a4 : ┤ rx ├──┼─────┼─────┼─────●─────┼─────●─────┼─────┼───────────────────────┼──────────┼─────┼─────●───────────┼─────┼─────●───┤ rx ├──┼─────┼───
        ╰─┬──╯  │     │     │     │     │     │     │     │                     ╭─┴─╮        │     │     │           │     │     │   ╰─┬──╯  │     │   
 __q3 : ──┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────────────────────┤ x ├────────┼─────┼─────┼───────────┼─────┼─────┼─────┼─────┼─────┼───
          │     │     │     │     │     │     │     │     │                     ╰───╯        │     │     │           │     │     │     │     │     │   
 __q2 : ──●─────◯─────◯─────●─────┼─────┼─────┼─────┼─────┼──────────────────────────────────┼─────◯─────┼───────────┼─────◯─────┼─────●─────┼─────●───
          │     │     │     │     │     │     │     │     │                                  │     │     │           │     │     │     │     │     │   
 __q1 : ──●─────◯─────●─────◯─────┼─────┼─────┼─────┼─────┼──────────────────────────────────┼─────●─────┼───────────┼─────◯─────┼─────●─────┼─────◯───
                                  │     │     │     │     │                                  │           │           │           │           │         
 __q0 : ──────────────────────────◯─────●─────●─────◯─────◯──────────────────────────────────◯───────────●───────────●───────────◯───────────◯─────────
                                                                                                                                                       
```
which is the same.

---------------------------------------------------------------
/external/mockturtle/mockturtle/networks/xag.hpp

implementation of the xag network.
