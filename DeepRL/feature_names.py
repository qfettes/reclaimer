feature_names = [
    'rx_packets', 'rx_bytex', 'tx_packets', 'tx_bytes',
    'rss', 'cache memory', 'page faults', 
    'cpu time',
    'io bytes', 'ret io serviced', 
    'rps', 'fps', 'request', 'failure',
    '50.0', '66.0','75.0','80.0','90.0',
    '95.0','98.0','99.0','99.9','99.99',
    '99.999','100.0', 
    'core cap', 
    'previous action',
    'QoS',
]
for i in range(29+19):
    feature_names.append(f'Microservice {i} Unique ID')

feature_names = feature_names + [x+'-1' for x in feature_names]
feature_names = feature_names + [x+'-2' for x in feature_names]
feature_names = feature_names + [x+'-3' for x in feature_names]
feature_names = feature_names + [x+'-4' for x in feature_names]