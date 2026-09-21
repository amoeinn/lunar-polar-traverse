#!/usr/bin/env bash
# Download the LOLA products, NAC comparison images and SPICE kernels into data/.
# Resumable: rerunning continues partial files and skips complete ones.
set -euo pipefail
cd "$(dirname "$0")/.."

PGDA=https://pgda.gsfc.nasa.gov/data/LOLA_20mpp
NAIF=https://naif.jpl.nasa.gov/pub/naif/generic_kernels

mkdir -p data/lola data/nac data/kernels

fetch() {
    wget -c -q --show-progress -P "$2" "$1"
}

# DEMs and PSR map (Barker et al. 2023, PGDA product 90)
for f in LDEM_80S_20MPP_ADJ.TIF LDEM_80S_80MPP_ADJ.TIF LPSR_80S_20MPP_ADJ.TIF; do
    fetch "$PGDA/$f" data/lola
done

# NAC images map-projected onto the 20 m LDEM, with their simulated counterparts
for id in M1188426614 M1311409938 M1312308500 M139356343 M139729686 \
          M151364517 M154940551 M155368056 M184288222; do
    for suffix in _20MPP_MODEL LE_20MPP RE_20MPP; do
        fetch "$PGDA/${id}${suffix}.TIF" data/nac
    done
done

# SPICE: DE421 throughout, matching the DEM's MOON_ME frame
fetch "$NAIF/lsk/naif0012.tls" data/kernels
fetch "$NAIF/pck/pck00010.tpc" data/kernels
fetch "$NAIF/pck/moon_pa_de421_1900-2050.bpc" data/kernels
fetch "$NAIF/spk/planets/de421.bsp" data/kernels \
    || fetch "$NAIF/spk/planets/a_old_versions/de421.bsp" data/kernels
fetch "$NAIF/fk/satellites/moon_080317.tf" data/kernels \
    || fetch "$NAIF/fk/satellites/a_old_versions/moon_080317.tf" data/kernels

echo "done"
ls -la data/lola data/nac data/kernels
