# dd_startup

## Aim of the project
The aim of this project is to create a **device-agnostic, integrated analysis toolbox** to **evaluate the operational regime that could enable a D-D startup** built upon open-source tools. 

The model couples plasma performance (density and temperature profiles), neutronics for D–D and D–T neutron spectra, transient fuel-cycle inventory dynamics, tritium processing times, and economic drivers. 

Two complementary approaches are used: (1) a **target-accumulation estimate**, computing the time to produce and store a startup inventory under the assumption of steady-state D-D operation, and (2) a** time-dependent model  of the  fuel-cycle inventories** to capture transient coupling between production, burn, and processing in the fuel cycle as the plasma is started in D-D and progressively transitioned up to a 50-50 D-T mixture by injecting the tritium produced on-site.

Rather than claiming a single best route, our aim is to **map the design space and quantify sensitivities across technical and financial levers**.

**These open, reproducible, and extensible analyses are intended to help designers, modelers, and decision-makers assess the feasibility, timelines, and economic requirements of tritium-free or tritium-lean startup strategies.**

## Physical background - Generating trtium in a fusion reactor
The main reactions inside a D-D mixture are:
D + D -> T + p
D + D -> He3 + n
D + T -> He4 + n
D + He3 -> He4 + p

The main pathways leading to trtium production are then:

            |----> He3 + n (2.45 MeV)
            |            |--------------------> Tdot_breedingDD: tritium production due to DD neutrons 
            |                                   interacting with the Li6 in the breeding blanket
    D + D ->|
      |     |----> T + p
      |            |
      |------------|---> D + T --> He4 + n (14.1 MeV)
                   |                     |----> Tdot_breedingDT: tritium production due to DT neutrons 
                   |                            interacting with the breeding blanket
                   |
                   |--------------------------> Tdot_fusion: tritium production due to DDp fusions, 
                                                considering the losses due to DT neutrons 

## Main parameters considered (PRELIMINARY VALUES)
The following parameters have been selected for the analysis (suggested maximum ranges are also reported):

| Parameter | Symbol | Unit | Range |
| --------- | ------ | ---- | ----- |
| Reactor parameters |
| --------- | ------ | ---- | ----- |
| Plasma volume | V_p | m<sup>3</sup> | 10 - 200 |
| Total ion density | n_i_tot | m<sup>-3</sup> | 10<sup>20</sup> - 10<sup>21</sup> |
| Ion temperature | T_i | keV | 10-100 |
| Tritium confinement time | tau<sub>p,T</sub> | s | 0.1 - 5 |
| Helium-3 confinement time | tau<sub>p,He3</sub> | s | 0.1 - 5 |
| Auxiliary power (D-D operation) | P<sub>aux</sub> | MW | 0 - 200 |
| Radiation losses (D-D operation) | P<sub>rad</sub> | MW | 0 - 100 |
| Auxiliary power (D-T operation) | P<sub>aux,DT</sub> | MW | 0 - 200 |
| Radiation losses (D-T operation) | P<sub>rad,DT</sub> | MW | 0 - 100 |
| -------- | -------- | -------- | -------- |
| Fuel cycle parameters |
| --------- | ------ | ---- | ----- |
| Tritium Breeding Ratio for D-T neutrons | TBR <sub>DT</sub> | - | 1.05 - 1.2 |
| Tritium Breeding Ratio for D-D neutrons | TBR <sub>DD</sub> | - | 0.5 - 1 |
| Inner fuel cycle residence time | $\tau$<sub>ifc</sub> | h | 1 - 24 |
| Inner fuel cycle residence time | $\tau$<sub>ifc</sub> | h | 12 - 48 |
| Target inventory | I<sub>target</sub> | kg | 0.1 - 5 |
| -------- | -------- | -------- | -------- |
| Plant and economic parameters |
| --------- | ------ | ---- | ----- |
| Thermal efficiency | $\eta_{th}$ | - | 0.3 - 0.5 |
| Plant availability | A | - | 0.5 - 0.9 |
| Cost of electricity | C<sub>el</sub> | $/kWh | 0.15 - 0.5 |


