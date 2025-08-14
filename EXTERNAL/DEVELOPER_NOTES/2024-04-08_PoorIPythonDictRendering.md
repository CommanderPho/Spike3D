## How it's poorly rendered by default:

- Lots of wasted vertical space, which is scarce on modern widescreen displays, and very poor use of the large horizontal space available.
- the values of the dict are not logically together, implying equal priority between the keys and arbitrary values.
- some value lists are broken after 1 item, other's two, based just on the number of characters.
- there's not clarifying indentation or anything on the arbitrarily broken lists
- I believe how it wraps could be font and screen-size dependent, meaning different programmers/users see different things
```python
{'known_named_decoding_epochs_type': ['laps',
  'pbe',
  'non_pbe',
  'global',
  'non_pbe_endcaps'],
 'trained_compute_epochs': ['non_pbe', 'laps'],
 'masked_time_bin_fill_type': ['ignore',
  'last_valid',
  'nan_filled',
  'dropped']}
```

![[Pasted image 20250408175159.png]]

### How I want it 
```python
{'known_named_decoding_epochs_type': ['laps', 'pbe', 'non_pbe', 'global', 'non_pbe_endcaps'],
 'trained_compute_epochs': ['non_pbe', 'laps'],
 'masked_time_bin_fill_type': ['ignore', 'last_valid', 'nan_filled', 'dropped'],
}
```

How can I make this happen consistently (for all output structures) in my Jupyter Python coding environment?