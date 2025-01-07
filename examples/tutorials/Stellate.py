from arborize import define_model

definitionStellate = define_model(
    {
        "synapse_types": {
            "AMPA": {
                "mechanism": "AMPA",
                "parameters": {
                    "tau_facil": 10.8,
                    "tau_rec": 35.1,
                    "tau_1": 10,
                    "gmax": 2300,
                    "U": 0.15,
                },
            },
            "NMDA": {
                "mechanism": ("NMDA", "stellate"),
                "parameters": {
                    "tau_facil": 5,
                    "tau_rec": 8,
                    "tau_1": 1,
                    "gmax": 10000,
                    "U": 0.15,
                },
            },
            "GABA": {
                "mechanism": "GABA",
                "parameters": {
                    "tau_facil": 0,
                    "tau_rec": 38.7,
                    "tau_1": 1,
                    "gmaxA1": 3230,
                    "U": 0.42,
                    "Erev": -65,
                },
            },
        },
        "cable_types": {
            "soma": {
                "cable": {"Ra": 110, "cm": 1},
                "ions": {
                    "na": {"rev_pot": 60},
                    "k": {"rev_pot": -84},
                    "ca": {"rev_pot": 137.5},
                    "h": {"rev_pot": -34},
                },
                "mechanisms": {
                    "Leak": {"e": -52, "gmax": 3e-05},
                    "Nav1_1": {"gbar": 0.2},
                    "Cav3_2": {"gcabar": 0.00163912063769},
                    "Cav3_3": {"pcabar": 1.615552993e-05},
                    "Kir2_3": {"gkbar": 1.093425575e-05},
                    "Kv1_1": {"gbar": 0.00107430134923},
                    "Kv3_4": {"gkbar": 0.008},
                    "Kv4_3": {"gkbar": 0.00404228168138},
                    "Kca1_1": {"gbar": 0.00518036298671},
                    "Kca2_2": {"gkbar": 0.00054166094878},
                    "Cav2_1": {"pcabar": 0.0005},
                    "HCN1": {"gbar": 0.00058451678362},
                    "cdp5": {"TotalPump": 7e-09},
                },
            },
            "dendrites": {"cable": {}, "ions": {}, "mechanisms": {}},
            "proximal_dendrites": {
                "cable": {"Ra": 110, "cm": 1.5},
                "ions": {"k": {"rev_pot": -84}, "ca": {"rev_pot": 137.5}},
                "mechanisms": {
                    "Leak": {"e": -48, "gmax": 8e-06},
                    "Cav3_2": {"gcabar": 0.00070661092763},
                    "Cav3_3": {"pcabar": 1.526216781e-05},
                    "Kv1_1": {"gbar": 0.0090681056165},
                    "Kv4_3": {"gkbar": 0.0026420471354},
                    "Kca1_1": {"gbar": 0.00499205404769},
                    "Kca2_2": {"gkbar": 3.26194117e-06},
                    "Cav2_1": {"pcabar": 0.0008},
                    "cdp5": {"TotalPump": 1e-09},
                },
                "synapses": {
                    "AMPA": {
                        "mechanism": "AMPA",
                        "attributes": {
                            "tau_facil": 10.8,
                            "tau_rec": 35.1,
                            "tau_1": 10,
                            "gmax": 2300,
                            "U": 0.15,
                        },
                    },
                    "NMDA": {
                        "mechanism": ("NMDA", "stellate"),
                        "attributes": {
                            "tau_facil": 5,
                            "tau_rec": 8,
                            "tau_1": 1,
                            "gmax": 10000,
                            "U": 0.15,
                        },
                    },
                },
            },
            "distal_dendrites": {
                "cable": {"Ra": 110, "cm": 1.5},
                "ions": {"k": {"rev_pot": -84}, "ca": {"rev_pot": 137.5}},
                "mechanisms": {
                    "Leak": {"e": -48, "gmax": 8e-06},
                    "Kv1_1": {"gbar": 0.00237825442906},
                    "Kca1_1": {"gbar": 0.00226329455766},
                    "Kca2_2": {"gkbar": 1.079984416e-05},
                    "Cav2_1": {"pcabar": 0.00025},
                    "cdp5": {"TotalPump": 1e-09},
                },
                "synapses": {
                    "AMPA": {
                        "mechanism": "AMPA",
                        "attributes": {
                            "tau_facil": 10.8,
                            "tau_rec": 35.1,
                            "tau_1": 10,
                            "gmax": 2300,
                            "U": 0.15,
                        },
                    },
                    "NMDA": {
                        "mechanism": ("NMDA", "stellate"),
                        "attributes": {
                            "tau_facil": 5,
                            "tau_rec": 8,
                            "tau_1": 1,
                            "gmax": 10000,
                            "U": 0.15,
                        },
                    },
                },
            },
            "axon": {
                "cable": {
                    "Ra": 110,
                    "cm": 1,
                },
                "ions": {
                    "na": {"rev_pot": 60},
                    "k": {"rev_pot": -84},
                    "h": {"rev_pot": -34},
                },
                "mechanisms": {
                    "Leak": {"e": -48, "gmax": 0.000008},
                    "Kv1_1": {"gbar": 0.00271359229578},
                    "Nav1_6": {"gbar": 0.00835931586458},
                    "Kv3_4": {"gkbar": 0.01153520393521},
                    "HCN1": {"gbar": 0.00070017344082},
                    "cdp5": {},
                },
            },
            "axon_initial_segment": {
                "cable": {"Ra": 110, "cm": 1},
                "ions": {
                    "na": {"rev_pot": 60},
                    "k": {"rev_pot": -84},
                    "h": {"rev_pot": -34},
                },
                "mechanisms": {
                    "Leak": {"e": -48, "gmax": 8e-06},
                    "HCN1": {"gbar": 0.00099184971498},
                    "Nav1_6": {"gbar": 0.3},
                    "Kv1_1": {"gbar": 0.00492841685426},
                    "Kv3_4": {"gkbar": 0.03351450571128},
                    "Km": {"gkbar": 7.960307413e-05},
                    "cdp5": {},
                },
            },
        },
    }
)

tagsSt = {
    16: ["dendrites", "proximal_dendrites"],
    17: ["dendrites", "distal_dendrites"],
    18: ["axon", "axon_initial_segment"],
}
