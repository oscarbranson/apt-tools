import pandas as pd
import struct

def read_pos(f):
    """ Loads an APT .pos file as a pandas dataframe.

    Columns:
        x: Reconstructed x position
        y: Reconstructed y position
        z: Reconstructed z position
        Da: mass/charge ratio of ion"""
    # read in the data
    n = len(file(f).read())/4
    d = struct.unpack('>'+'f'*n,file(f).read(4*n))
                    # '>' denotes 'big-endian' byte order
    # unpack data
    pos = pd.DataFrame({'x': d[0::4],
                        'y': d[1::4],
                        'z': d[2::4],
                        'Da': d[3::4]})
    return pos


def read_epos(f):
    """Loads an APT .epos file as a pandas dataframe.

    Columns:
        x: Reconstructed x position
        y: Reconstructed y position
        z: Reconstructed z position
        Da: Mass/charge ratio of ion
        ns: Ion Time Of Flight
        DC_kV: Potential
        pulse_kV: Size of voltage pulse (voltage pulsing mode only)
        det_x: Detector x position
        det_y: Detector y position
        pslep: Pulses since last event pulse (i.e. ionisation rate)
        ipp: Ions per pulse (multihits)

     [x,y,z,Da,ns,DC_kV,pulse_kV,det_x,det_y,pslep,ipp].
        pslep = pulses since last event pulse
        ipp = ions per pulse

    When more than one ion is recorded for a given pulse, only the
    first event will have an entry in the "Pulses since last evenT
    pulse" column. Each subsequent event for that pulse will have
    an entry of zero because no additional pulser firings occurred
    before that event was recorded. Likewise, the "Ions Per Pulse"
    column will contain the total number of recorded ion events for
    a given pulse. This is normally one, but for a sequence of records
    a pulse with multiply recorded ions, the first ion record will
    have the total number of ions measured in that pulse, while the
    remaining records for that pulse will have 0 for the Ions Per
    Pulse value.
        ~ Appendix A of 'Atom Probe tomography: A Users Guide',
          notes on ePOS format."""
    # read in the data
    n = len(file(f).read())/4
    rs = n / 11
    d = struct.unpack('>'+'fffffffffII'*rs,file(f).read(4*n))
                    # '>' denotes 'big-endian' byte order
    # unpack data
    pos = pd.DataFrame({'x': d[0::11],
                        'y': d[1::11],
                        'z': d[2::11],
                        'Da': d[3::11],
                        'ns': d[4::11],
                        'DC_kV': d[5::11],
                        'pulse_kV': d[6::11],
                        'det_x': d[7::11],
                        'det_y': d[8::11],
                        'pslep': d[9::11], # pulses since last event pulse
                        'ipp': d[10::11]}) # ions per pulse
    return pos


def read_rrng(f):
    """Loads a .rrng file produced by IVAS. Returns two dataframes of 'ions'
    and 'ranges'."""
    import re

    rf = open(f,'r').readlines()

    patterns = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')

    ions = []
    rrngs = []
    for line in rf:
        m = patterns.search(line)
        if m:
            if m.groups()[0] is not None:
                ions.append(m.groups()[:2])
            else:
                rrngs.append(m.groups()[2:])

    ions = pd.DataFrame(ions, columns=['number','name'])
    ions.set_index('number',inplace=True)
    rrngs = pd.DataFrame(rrngs, columns=['number','lower','upper','vol','comp','colour'])
    rrngs.set_index('number',inplace=True)

    rrngs[['lower','upper','vol']] = rrngs[['lower','upper','vol']].astype(float)
    rrngs[['comp','colour']] = rrngs[['comp','colour']].astype(str)

    return ions,rrngs


def label_ions(pos,rrngs):
    """labels ions in a .pos or .epos dataframe (anything with a 'Da' column)
    with composition and colour, based on an imported .rrng file."""

    pos['comp'] = ''
    pos['colour'] = '#FFFFFF'

    for n,r in rrngs.iterrows():
        pos.loc[(pos.Da >= r.lower) & (pos.Da <= r.upper),['comp','colour']] = [r['comp'],'#' + r['colour']]

    return pos


def deconvolve(lpos):
    """Takes a composition-labelled pos file, and deconvolves
    the complex ions. Produces a dataframe of the same input format
    with the extra columns:
       'element': element name
       'n': stoichiometry
    For complex ions, the location of the different components is not
    altered - i.e. xyz position will be the same for several elements."""

    import re

    out = []
    pattern = re.compile(r'([A-Za-z]+):([0-9]+)')

    for g,d in lpos.groupby('comp'):
        if g is not '':
            for i in range(len(g.split(' '))):
                tmp = d.copy()
                cn = pattern.search(g.split(' ')[i]).groups()
                tmp['element'] = cn[0]
                tmp['n'] = cn[1]
                out.append(tmp.copy())
    return pd.concat(out)
