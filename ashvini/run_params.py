from pathlib import Path
from dataclasses import dataclass, asdict
import yaml


@dataclass
class IOParams:
    mass_bin: float
    tree_file: str
    dir_out: str


@dataclass
class StarFormationParams:
    efficiency: float


@dataclass
class SupernovaParams:
    type: str
    delay_time: float
    epsilon_p: float
    pi_fid: float


@dataclass
class ReionizationParams:
    UVB_enabled: bool
    z_reion: float
    gamma: float
    omega: float


@dataclass
class MetalsParams:
    Z_IGM: float
    Z_yield: float


@dataclass
class Params:
    sf: StarFormationParams
    sn: SupernovaParams
    reion: ReionizationParams
    metals: MetalsParams
    io: IOParams


def load_params(verbose=True) -> Params:
    root = Path(__file__).resolve().parents[1]
    config_file = root / "run_params.yaml"
    raw = yaml.safe_load(config_file.read_text())

    params = Params(
        sf=StarFormationParams(**raw["star_formation"]),
        sn=SupernovaParams(**raw["supernova"]),
        reion=ReionizationParams(**raw["reionization"]),
        metals=MetalsParams(**raw["metallicity"]),
        io=IOParams(**raw["basics"]),
    )

    return params


def print_config(params: Params):
    print("\n Loaded simulation parameters:\n")

    def section(title, d):
        print(f"\n[{title}]")
        for k, v in d.items():
            print(f"  {k:<16} : {v}")

    section("Basics", asdict(params.io))
    section("Star Formation", asdict(params.sf))
    section("Supernova Feedback", asdict(params.sn))
    section("Reionization", asdict(params.reion))
    section("Metals", asdict(params.metals))


PARAMS = load_params()
