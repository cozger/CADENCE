"""Check actual AU channel names from XDF stream metadata."""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')
import pyxdf, glob

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

for stream in data:
    name = stream['info']['name'][0]
    if 'landmarks' in name.lower():
        print(f"Stream: {name}")
        n_ch = int(stream['info']['channel_count'][0])
        print(f"  Channels: {n_ch}")
        # Check for channel descriptions in metadata
        desc = stream['info'].get('desc', [{}])
        if desc and desc[0]:
            channels = desc[0].get('channels', [{}])
            if channels and channels[0]:
                ch_list = channels[0].get('channel', [])
                if ch_list:
                    print(f"  Channel names ({len(ch_list)} found):")
                    for i, ch in enumerate(ch_list[:60]):
                        label = ch.get('label', ['?'])[0] if isinstance(ch.get('label'), list) else ch.get('label', '?')
                        print(f"    [{i:3d}] {label}")
                else:
                    print("  No channel list in desc")
            else:
                print("  No channels in desc")
        else:
            print("  No desc metadata")
        print()
