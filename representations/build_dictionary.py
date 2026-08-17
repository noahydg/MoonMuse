from events2words import events2dictionary
import pickle

if __name__ == '__main__':
    out_d = {'hooktheory_chord': r'events/chord/hooktheory_chord_functional/events',
              'emopia_chord': r'events/chord/emopia_chord_functional/events',
              'pop1k7_chord': r'events/chord/pop1k7_chord_functional/events',
              'emopia_c2m': r'events/c2m/emopia_c2m_functional/events',
              'pop1k7_c2m': r'events/c2m/pop1k7_c2m_functional/events',
              'emopia_m2p': r'events/m2p/emopia_m2p_functional/events',
              'pop1k7_m2p': r'events/m2p/pop1k7_m2p_functional/events',
              'emopia_raw': r'events/raw/emopia_raw_functional/events',
              'pop1k7_raw': r'events/raw/pop1k7_raw_functional/events',
              'emopia_fill': r'events/fill/emopia_fill_functional/events',
              'pop1k7_fill': r'events/fill/pop1k7_fill_functional/events'}

    
    out_d = {k: v.replace('/events', '') for k, v in out_d.items()}

    # events2dictionary(out_d['hooktheory_chord'], add_chord=True, check=True, relative=None, 
    #                   event_pos=1, num_emotion=2, add_emotion=True, add_tempo=False, add_velocity=False)
    # events2dictionary(out_d['emopia_chord'], add_chord=True, check=True, relative=None, 
    #                   event_pos=1, num_emotion=2, add_emotion=True, add_tempo=False, add_velocity=False)
    
    # events2dictionary(out_d['emopia_c2m'], add_chord=False, check=False, relative=True, 
    #                   event_pos=2, num_emotion=4, add_emotion=True, add_tempo=False, add_velocity=False)
    # events2dictionary(out_d['pop1k7_c2m'], add_chord=False, check=False, relative=True, 
    #                   event_pos=2, num_emotion=4, add_emotion=True, add_tempo=False, add_velocity=False)
    
    # events2dictionary(out_d['emopia_m2p'], add_chord=False, check=False, relative=True, 
    #                   event_pos=2, num_emotion=4, add_emotion=True, add_tempo=True, add_velocity=True)
    # events2dictionary(out_d['pop1k7_m2p'], add_chord=False, check=False, relative=True, 
    #                   event_pos=2, num_emotion=4, add_emotion=True, add_tempo=True, add_velocity=True)
    
    # events2dictionary(out_d['emopia_raw'], add_chord=True, check=False, relative=True, 
    #                   event_pos=1, num_emotion=4, add_emotion=True, add_tempo=True, add_velocity=True,
    #                   add_mask=True, 
    #                   prefix_list=['Note_Velocity', 'Tempo'],
    #                   exclude=True
    #                   )
    # events2dictionary(out_d['pop1k7_raw'], add_chord=True, check=False, relative=True, 
    #                   event_pos=1, num_emotion=4, add_emotion=True, add_tempo=True, add_velocity=True,
    #                   add_mask=True, 
    #                   prefix_list=['Note_Velocity', 'Tempo'],
    #                   exclude=True
    #                   )
    events2dictionary(out_d['pop1k7_fill'], add_chord=True, check=False, relative=True, 
                    event_pos=1, num_emotion=4, add_emotion=True, add_tempo=True, add_velocity=True,
                    add_mask=True, 
                    prefix_list=['Note_Velocity', 'Tempo', 'Note_Duration'],
                    exclude=True
                    )
    events2dictionary(out_d['emopia_fill'], add_chord=True, check=False, relative=True, 
                    event_pos=1, num_emotion=4, add_emotion=True, add_tempo=True, add_velocity=True,
                    add_mask=True, 
                    prefix_list=['Note_Velocity', 'Tempo', 'Note_Duration'],
                    exclude=True          
                    )

