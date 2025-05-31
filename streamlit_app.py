# 3. Tampilkan PBest dan GBest
st.subheader("Personal Best (PBest) dan Global Best (GBest)")

# Tampilkan GBest
st.markdown(f"""
**Global Best (GBest):**
- Gamma Optimal: `{best_gamma:.6f}`
- Fitness Value: `{best_cost:.6f}`
""")

# Tampilkan PBest dalam bentuk tabel
st.markdown("**Personal Best (PBest) dari semua partikel:**")

# Buat dataframe untuk PBest
pbest_data = []
for i, (pos, cost) in enumerate(zip(optimizer.swarm.pbest_pos, optimizer.swarm.pbest_cost)):
    pbest_data.append({
        'Partikel': i+1,
        'Gamma': pos[0],
        'Fitness': cost
    })

pbest_df = pd.DataFrame(pbest_data)
st.dataframe(
    pbest_df.style.format({
        'Gamma': '{:.6f}',
        'Fitness': '{:.6f}'
    }),
    height=400,
    use_container_width=True
)
