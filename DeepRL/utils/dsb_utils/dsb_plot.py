import plotly.express as px
import pandas as pd
import os


from utils.dsb_utils.dsb_ctrl import _get_int_val, _get_float_val

def plot_stress_test_data(dir):
    end_to_end_lat = {}
    end_to_end_lat['users'] = []
    end_to_end_lat['50.0'] = []
    end_to_end_lat['66.0'] = []
    end_to_end_lat['75.0'] = []
    end_to_end_lat['80.0'] = []
    end_to_end_lat['90.0'] = []
    end_to_end_lat['95.0'] = []
    end_to_end_lat['98.0'] = []
    end_to_end_lat['99.0'] = []
    end_to_end_lat['99.9'] = []
    end_to_end_lat['99.99'] = []
    end_to_end_lat['99.999'] = []
    end_to_end_lat['100.0'] = []
    end_to_end_lat['fps'] = []
    end_to_end_lat['rps'] = []


    for subdir, dirs, _ in os.walk(dir):
        padded_dirs = []
        for dir in dirs:
            split_str = dir.split('_')
            split_str[-1] = split_str[-1].zfill(3)
            padded_dirs.append('_'.join(split_str))
            os.rename(os.path.join(subdir, dir), os.path.join(subdir, padded_dirs[-1]))
            
        
        for dir in sorted(padded_dirs):
            fname = os.path.join(subdir, dir, 'social_stats.csv')
            assert(os.path.exists(fname)), f'stress test result {fname} doesn\'t exist!'

            with open(fname, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 1
                fields = lines[0].split(',')

                pos = {}
                pos['50%'] = None
                pos['66%'] = None
                pos['75%'] = None
                pos['80%'] = None
                pos['90%'] = None
                pos['95%'] = None
                pos['98%'] = None
                pos['99%'] = None
                pos['99.9%'] = None
                pos['99.99%'] = None
                pos['99.999%'] = None
                pos['100%'] = None
                pos['rps'] = None
                pos['fps'] = None
                
                for i, k in enumerate(fields):
                    k = k.replace('\"', '').strip()
                    if k == '50%':
                        pos['50%'] = i
                    elif k == '66%':
                        pos['66%'] = i
                    elif k == '75%':
                        pos['75%'] = i
                    elif k == '80%':
                        pos['80%'] = i
                    elif k == '90%':
                        pos['90%'] = i
                    elif k == '95%':
                        pos['95%'] = i
                    elif k == '98%':
                        pos['98%'] = i
                    elif k == '99%':
                        pos['99%'] = i
                    elif k == '99.9%':
                        pos['99.9%'] = i
                    elif k == '99.99%':
                        pos['99.99%'] = i
                    elif k == '99.999%':
                        pos['99.999%'] = i
                    elif k == '100%':
                        pos['100%'] = i
                    elif k == 'Requests/s':
                        pos['rps'] = i
                    elif k == 'Failures/s':
                        pos['fps'] = i

                data = lines[-1].split(',')

                end_to_end_lat['users'].append( _get_int_val(dir.split('_')[-1]) )
                end_to_end_lat['fps'].append( _get_float_val(data[ pos['fps'] ]) ) # failures/s
                end_to_end_lat['rps'].append( _get_float_val(data[ pos['rps'] ]) ) # requests/s
                end_to_end_lat['50.0'].append( _get_int_val(data[ pos['50%'] ]) )
                end_to_end_lat['66.0'].append( _get_int_val(data[ pos['66%'] ]) )
                end_to_end_lat['75.0'].append( _get_int_val(data[ pos['75%'] ]) )
                end_to_end_lat['80.0'].append( _get_int_val(data[ pos['80%'] ]) )
                end_to_end_lat['90.0'].append( _get_int_val(data[ pos['90%'] ]) )
                end_to_end_lat['95.0'].append( _get_int_val(data[ pos['95%'] ]) )
                end_to_end_lat['98.0'].append( _get_int_val(data[ pos['98%'] ]) )
                end_to_end_lat['99.0'].append( _get_int_val(data[ pos['99%'] ]) )
                end_to_end_lat['99.9'].append( _get_int_val(data[ pos['99.9%'] ]) )
                end_to_end_lat['99.99'].append( _get_int_val(data[ pos['99.99%'] ]) )
                end_to_end_lat['99.999'].append( _get_int_val(data[ pos['99.999%'] ]) )
                end_to_end_lat['100.0'].append( _get_int_val(data[ pos['100%'] ]) )

    data_preproc = pd.DataFrame(end_to_end_lat)

    y_data = '99.0'
    fig = px.line(data_preproc, y=y_data, x='users', title='All Data')
    y_data.replace('.', '-')
    fig.write_image(f'{y_data}_tail_latency'.replace('.', '_') + '.png')

if __name__=='__main__':
    rootdir = '../Data/stress_test_data/'
    plot_stress_test_data(rootdir)