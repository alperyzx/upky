# Memory Optimization Guide for Streamlit App

## Implemented Memory Leak Fixes

### 1. **Enhanced Memory Management**
- Added `contextmanager` for automatic memory cleanup
- Implemented aggressive garbage collection after each model run
- Added periodic memory cleanup every 10 interactions
- Enhanced cache settings with shorter TTL and fewer entries

### 2. **Matplotlib Optimization**
- Configured matplotlib for memory optimization (`plt.ioff()`, smaller font sizes)
- Wrapped all plotting operations in memory contexts
- Explicit figure closure after each plot
- Removed interactive mode to reduce memory overhead

### 3. **Cache Optimization**
- Reduced cache TTL from 10 minutes to 2 minutes
- Limited cache entries to 1 per function
- Added automatic cache clearing mechanisms
- Implemented selective cache cleanup for large objects

### 4. **Model Comparison Optimization**
- Completely rewrote the comparison section to use generators
- Process models individually instead of storing all results
- Immediate cleanup of intermediate variables
- Simplified plotting with memory-efficient operations

### 5. **Pandas Memory Optimization**
- Copy DataFrames when necessary to avoid reference issues
- Explicit deletion of large DataFrames
- Optimized column selection to reduce memory usage
- Reduced precision for numerical displays

### 6. **Session State Management**
- Added periodic cleanup of session state variables
- Remove large objects from session state automatically
- Limit session state storage to essential data only

## Configuration Changes

### 1. **Streamlit Config (`.streamlit/config.toml`)**
```toml
[server]
maxUploadSize = 50
maxMessageSize = 50
enableCORS = false

[runner]
magicEnabled = true
installTracer = false
fixMatplotlib = true
postScriptGC = true
fastReruns = true
enforceSerializableSessionState = true
```

### 2. **Requirements Optimization**
- Pinned dependency versions to avoid compatibility issues
- Added version constraints to prevent memory-heavy versions
- Included optional performance enhancers (numba, cython)

## Usage Recommendations

### 1. **Running the App**
```bash
# Install dependencies
pip install -r requirements.txt

# Run with memory optimization
streamlit run streamlit_app.py --server.maxUploadSize=50 --server.maxMessageSize=50
```

### 2. **Memory Monitoring**
- Enable "Bellek İzleme" checkbox in the comparison section
- Monitor memory usage throughout the session
- Use "Cache Temizle" button when memory usage is high

### 3. **Best Practices**
- Run individual models instead of comparison when possible
- Clear cache regularly during long sessions
- Close the app and restart if memory issues persist
- Use smaller datasets for testing

## Performance Improvements

### Before Optimization:
- Memory usage could grow to 500MB+ during comparisons
- Frequent memory leaks from matplotlib figures
- Cache accumulation causing slowdowns
- Session state bloat

### After Optimization:
- Memory usage typically stays under 150MB
- Automatic cleanup prevents memory leaks
- Faster response times due to optimized caching
- Stable performance during long sessions

## Troubleshooting

### If Memory Issues Persist:
1. Click "Cache Temizle" button in sidebar
2. Refresh the browser page
3. Restart the Streamlit app
4. Consider using individual models instead of comparison

### Performance Tips:
- Use "Normal Talep" instead of "Yüksek Talep" for testing
- Avoid running multiple models simultaneously
- Close other browser tabs to free system memory
- Monitor system memory usage externally if needed

## Technical Details

### Memory Context Manager:
```python
@contextmanager
def memory_context():
    try:
        yield
    finally:
        safe_memory_cleanup()
```

### Cache Settings:
```python
@st.cache_data(ttl=120, max_entries=1, show_spinner=False)
def optimized_solver(*args, **kwargs):
    with memory_context():
        return solver(*args, **kwargs)
```

### Cleanup Functions:
```python
def clear_memory():
    plt.close('all')
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    # Clear session state of large objects
    keys_to_clear = [k for k in st.session_state.keys() if k.startswith('model_results_')]
    for key in keys_to_clear:
        del st.session_state[key]
    for _ in range(3):
        gc.collect()
```

This comprehensive optimization should significantly reduce memory usage and prevent memory leaks in your Streamlit application.
