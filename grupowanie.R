library(dbscan)
library(fpc)
library(cluster)
library(mclust)
library(factoextra)
library(dplyr)
library(ggplot2)

download.file('https://staff.elka.pw.edu.pl/~rbembeni/dane/Pokemon.csv','Pokemon.csv')
pokemon <- read.csv("Pokemon.csv")

head(pokemon)
summary(pokemon)
str(pokemon)
View(pokemon)

sum(is.na(pokemon))

powers = pokemon[6:11]
View(powers)
summary(powers)

powers_S <- scale(powers, center = FALSE)
powers_S <- as.data.frame(powers_S)
summary(powers_S)
View(powers_S)

#==================================================================================================================

# 2. Grupowanie algorytmem partycjonującym:
# Wyznaczenie liczby grup dla algorytmu k-środków metodą „łokcia”

calculate_wss <- function(data, max_clusters) {
  wss <- numeric(max_clusters)
  for (i in 1:max_clusters) {
    km.out <- kmeans(data, centers = i, nstart = 20)
    wss[i] <- km.out$tot.withinss
  }
  return(wss)
}

# Losowanie próbki 25% danych
set.seed(123) 
sample_indices <- sample(1:nrow(powers_S), size = 0.25 * nrow(powers_S))
powers_sample <- powers_S[sample_indices, ]

summary(powers_sample)
View(powers_sample)
head(powers_sample)

# Obliczanie WSS
wss_1 <- calculate_wss(powers_sample, 20)

# Inny przykład:
set.seed(456) 
sample_indices2 <- sample(1:nrow(powers_S), size = 0.25 * nrow(powers_S))
powers_sample2 <- powers_S[sample_indices2, ]

wss_2 <- calculate_wss(powers_sample2, 30)


# Znalezienie punktu „łokcia”
find_elbow <- function(wss) {
  # Pierwsza pochodna (zmiana między kolejnymi punktami)
  first_diff <- diff(wss)
  # Druga pochodna (zmiana nachylenia)
  second_diff <- diff(first_diff)
  # Największa zmiana nachylenia)
  elbow <- which.max(abs(second_diff)) + 1 
  return(elbow)
}

elbow_point_1 <- find_elbow(wss_1)
elbow_point_2 <- find_elbow(wss_2)


# Wizualizacja WSS
par(mfrow = c(1, 2))
plot(1:20, wss_1, type = "b", 
     xlab = "Liczba grup (k)", 
     ylab = "Suma błędów kwadratowych wewnątrz grup (WSS)",
     main = "Metoda łokcia (Próba 1)")
points(elbow_point_1, wss_1[elbow_point_1], col = "red", pch = 19, cex = 1.5)
text(elbow_point_1, wss_1[elbow_point_1], labels = paste("k =", elbow_point_1), pos = 3, col = "red")
grid()

plot(1:30, wss_2, type = "b", 
     xlab = "Liczba grup (k)", 
     ylab = "Suma błędów kwadratowych wewnątrz grup (WSS)",
     main = "Metoda łokcia (Próba 2)")
points(elbow_point_2, wss_2[elbow_point_2], col = "red", pch = 19, cex = 1.5)
text(elbow_point_2, wss_2[elbow_point_2], labels = paste("k =", elbow_point_2), pos = 3, col = "red")
grid()


# Na podstawie wykresów wybrano 2 grupy.



# Wykonanie grupowania z różnymi wartościami parametrów

# 1. K-średnich z innymi miarami odległości 
cat("\nZastosowanie miary odległości: euclidean")
km_result_eu <- eclust(powers_sample, "kmeans", k = 2, hc_metric = "euclidean", graph = FALSE)
cat("\nZastosowanie miary odległości: manhattan")
km_result_man <- eclust(powers_sample, "kmeans", k = 2, hc_metric = "manhattan", graph = FALSE)

# Wizualizacja wyników klastrowania
fviz_cluster(km_result_eu, 
             geom = "point",
             main = paste("K-średnich z miarą odległości: euclidean"),
             palette = "jco")
fviz_cluster(km_result_man, 
             geom = "point",
             main = paste("K-średnich z miarą odległości: manhattan"),
             palette = "jco")


# 2. K-medoids (PAM) z różnymi miarami odległości
set.seed(123)
pam_out_man <- pam(powers_sample, k = 2, metric = "manhattan")
fviz_cluster(pam_out_man, data = powers_sample, ellipse.type = "norm", main = "Grupowanie k-medoids z miarą Manhattan")
pam_out_eu <- pam(powers_sample, k = 2, metric = "euclidean")
fviz_cluster(pam_out_eu, data = powers_sample, ellipse.type = "norm", main = "Grupowanie k-medoids z miarą Euclidean")


cat("\nZastosowanie miary odległości EUCLIDEAN:")
dist_matrix_e <- dist(powers_sample, "euclidean")
dist_matrix_e <- as.matrix(dist_matrix_e)
pam_result_e <- pam(dist_matrix_e, k = 2)

cat("\nZastosowanie miary odległości MANHATTAN:")
dist_matrix_mh <- dist(powers_sample, "manhattan")
dist_matrix_mh <- as.matrix(dist_matrix_mh)
pam_result_mh <- pam(dist_matrix_mh, k = 2)

cat("\nZastosowanie miary odległości MAXIMUM:")
dist_matrix_mx <- dist(powers_sample, "maximum")
dist_matrix_mx <- as.matrix(dist_matrix_mx)
pam_result_mx <- pam(dist_matrix_mx, k = 2)


# Wizualizacja wyników klastrowania
fviz_cluster(pam_result_e, 
             geom = "point",
             main = paste("PAM z miarą odległości: euclidean"),
             palette = "jco")
fviz_cluster(pam_result_mh, 
             geom = "point",
             main = paste("PAM z miarą odległości: manhattan"),
             palette = "jco")  # Wizualizacja wyników klastrowania
fviz_cluster(pam_result_mx, 
             geom = "point",
             main = paste("PAM z miarą odległości: maximum"),
             palette = "jco")


# 3. Hierarchiczne klastrowanie z miarą odległości CANBERRA:
cat("\nZastosowanie miary odległości: canberra")
dist_matrix <- dist(powers_sample, method = "canberra")
hc_result <- hclust(dist_matrix, method = "ward.D2")

fviz_dend(hc_result, 
          main = paste("Hierarchiczne klastrowanie z miarą odległości:", method),
          k = 2, 
          rect = TRUE)




# Ocena jakości grupowa przy użyciu indeksu Silhouette.

# Wykres oceny liczby klastrów dla PAM przy użyciu indeksu Silhouette
fviz_nbclust(powers_sample, pam, method = "silhouette")+
  ggtitle("Ocena liczby klastrów dla PAM za pomocą indeksu Silhouette") +
  theme_classic()

# Klastrowanie za pomocą PAM
pam.res <- pam(powers_sample, 2)
print(pam.res)
powers_sample_clus<-cbind(powers_sample, pam.res$cluster)
head(powers_sample_clus)
print(pam.res$medoids)
print(pam.res$clusinfo)
pam.res$clustering

# Wizualizacja klastrowania PAM
fviz_cluster(pam.res,
             palette = c("#00AFBB", "#FC4E07"),
             ellipse.type = "t", 
             repel = TRUE, 
             ggtheme = theme_light()
)

# Klastrowanie k-means:
set.seed(123)
km_out <- kmeans(powers_sample, centers = 2, nstart = 20)
silhouette_km <- silhouette(km_out$cluster, dist(powers_sample))
avg_silhouette <- mean(silhouette_km[, 3])
avg_silhouette
fviz_silhouette(silhouette_km) +
  ggtitle("Indeks Silhouette dla k-means") +
  annotate("text", x = 50, y = max(silhouette_km[, 3]),
           label = paste("Index Silhouette: ", round(avg_silhouette, 2)),
           size = 5, color = "blue")

# Klastrowanie k-medoids (pam):
pam_out <- pam(powers_sample, k = 2)
silhouette_pam <- silhouette(pam_out$cluster, dist(powers_sample))
avg_silhouette_p <- mean(silhouette_pam[, 3])
fviz_silhouette(silhouette_pam) +
  ggtitle("Indeks Silhouette dla k-medoids") +
  annotate("text", x = 50, y = max(silhouette_km[, 3]), 
           label = paste("Index Silhouette: ", round(avg_silhouette_p, 2)), 
           size = 5, color = "blue")

# Klastrowanie za pomocą eclust (k-means):
km_alt<-eclust(powers_sample, "kmeans", k=2, graph=TRUE)
fviz_silhouette(km_alt, palette="jco")
cl_stats_km <- cluster.stats(d = dist(powers_sample), km_alt$cluster)

# Klastrowanie za pomocą eclust (k-medoids)
pam.res <- eclust(powers_sample, "pam", k = 2, graph = TRUE)
fviz_silhouette(pam.res, palette = "jco")
cl_stats_pam<-cluster.stats(d = dist(powers_sample), pam.res$cluster)

cat("Średnia wartość indeksu Silhouette dla k-means:", round(avg_silhouette, 2), "\n")
cat("Średnia wartość indeksu Silhouette dla k-medoids:", round(avg_silhouette_p, 2), "\n")
cat("Średnia wartość indeksu Silhouette dla eclust (k-means):", round(avg_silhouette, 2), "\n")
cat("Średnia wartość indeksu Silhouette dla eclust (k-medoids):", round(avg_silhouette_p, 2), "\n")
cat("Indeks Silhouette dla eclust (k-means):", round(cl_stats_km$sindex, 2), "\n")
cat("Indeks Silhouette dla eclust (k-medoids):", round(cl_stats_pam$sindex, 2), "\n")




# Przypisanie poszczególnych rekordów do grup:

# Przypisanie rekordów do grup k-means - k-medoids
powers_sample$cluster_kmeans <- km_out$cluster
powers_sample$cluster_pam <- pam_out$cluster

# Wyświetlanie pierwszych kilku rekordów z przypisanymi grupami
head(powers_sample)

cat("Liczba rekordów w poszczególnych grupach (k-means):\n")
print(table(powers_sample$cluster_kmeans))
cat("Liczba rekordów w poszczególnych grupach (k-medoids):\n")
print(table(powers_sample$cluster_pam))

# Dodanie przypisanych grup do oryginalnego zbioru danych
pokemon$Cluster_kmeans <- km_out$cluster
head(pokemon[, c("Name", "Cluster_kmeans")])
pokemon$Cluster_PAM <- pam.res$cluster
head(pokemon[, c("Name", "Cluster_PAM")])

fviz_cluster(list(data = powers_sample, cluster = km_out$cluster),
             geom = "point",
             palette = "jco",
             main = "Wizualizacja grup k-średnich")

fviz_cluster(pam.res,
             geom = "point",
             palette = "jco",
             main = "Wizualizacja grup PAM")


# Grupowanie po klastrach i obliczenie średnich wartości cech
char_kmeans <- pokemon %>%
  group_by(Cluster_kmeans) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE))
print(char_kmeans)

char_pam <- pokemon %>%
  group_by(Cluster_PAM) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE))
print(char_pam)

# Wykres dla k-średnich
char_kmeans_long <- tidyr::pivot_longer(char_kmeans, cols = -Cluster_kmeans, names_to = "Feature", values_to = "Mean")

ggplot(char_kmeans_long, aes(x = Feature, y = Mean, fill = factor(Cluster_kmeans))) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Średnie wartości cech w grupach k-średnich", fill = "Grupa")




# Grupowanie algorytmem DBSCAN

# Wyznaczenie parametru eps dla algorytmu DBSCAN metodą szukania punktu przegięcie
dbscan::kNNdistplot(powers_sample, k=5)
abline(h=0.7, lty="dashed")

# Wykonanie grupowania dla kilku zestawów wartości parametrów.
# Parametry DBSCAN
eps_values <- c(0.5, 0.6, 0.7, 0.75)
minPts_values <- c(3, 5, 7, 10)

best_silhouette <- -Inf 
best_eps <- NULL
best_minPts <- NULL

for (eps in eps_values) {
  for (minPts in minPts_values) {
    dbscan_res <- dbscan::dbscan(powers_sample, eps = eps, minPts = minPts)
    silhouette_res <- silhouette(dbscan_res$cluster, dist(powers_sample))
    avg_silhouette <- mean(silhouette_res[, 3])
    cat("Indeks Silhouette dla eps =", eps, "i minPts =", minPts, ":", avg_silhouette, "\n")

    if (avg_silhouette > best_silhouette) {
      best_silhouette <- avg_silhouette
      best_eps <- eps
      best_minPts <- minPts
    }
  }
}

cat("Najlepsze parametry DBSCAN:\n")
cat("eps =", best_eps, ", minPts =", best_minPts, "\n")
powers_sample.dbscanBEST <- dbscan::dbscan(powers_sample, eps=0.7, minPts=5)

fviz_cluster(powers_sample.dbscanBEST, data = powers_sample, geom = "point", ellipse.type = "t") +
  ggtitle(paste("DBSCAN eps =", eps, ", minPts =", minPts))


# Ocena jakości grupowa przy użyciu indeksu Silhouette.
pokemon$Cluster_DBSCAN <- powers_sample.dbscanBEST$cluster
powers_sample_no_noise <- powers_sample[powers_sample.dbscanBEST$cluster != 0, ]
clusters_no_noise <- powers_sample.dbscanBEST$cluster[powers_sample.dbscanBEST$cluster != 0]

sil_dbscan <- silhouette(clusters_no_noise, dist(powers_sample_no_noise))
avg_silhouette_dbscan <- mean(sil_dbscan[, 3])
fviz_silhouette(sil_dbscan)
cat("Średnia wartość indeksu Silhouette dla BDSCAN:", round(avg_silhouette_dbscan, 2), "\n")


# Przypisanie poszczególnych rekordów do grup
pokemon$Cluster_DBSCAN <- powers_sample.dbscanBEST$cluster
cat("Liczba rekordów w poszczególnych grupach DBSCAN:\n")
print(table(pokemon$Cluster_DBSCAN))


# Znalezienie charakterystycznych elementów grup
group_summary <- pokemon %>%
  filter(Cluster_DBSCAN != 0) %>%  
  group_by(Cluster_DBSCAN) %>%
  summarise(
    Avg_HP = mean(HP),
    Avg_Attack = mean(Attack),
    Avg_Defense = mean(Defense),
    Avg_Speed = mean(Speed),
    Most_Common_Type = names(which.max(table(Type.1)))
  )
print(group_summary)

ggplot(pokemon, aes(x = HP, y = Attack, color = as.factor(Cluster_DBSCAN))) +
  geom_point() +
  labs(color = "Cluster_DBSCAN") +
  theme_minimal()



# Porównanie wyników uzyskanych dwoma metodami grupowania
cat("Porównanie liczby rekordów w grupach k-means i k-medoids:\n")
table(powers_sample$cluster_kmeans)
table(powers_sample$cluster_pam)

cat("Średnie wartości cech w grupach k-means:\n")
print(char_kmeans)
cat("Średnie wartości cech w grupach k-medoids:\n")
print(char_pam)

cat("Średnia wartość indeksu Silhouette dla k-means:", round(avg_silhouette, 2), "\n")
cat("Średnia wartość indeksu Silhouette dla k-medoids:", round(avg_silhouette_p, 2), "\n")
cat("Średnia wartość indeksu Silhouette dla BDSCAN:", round(avg_silhouette_dbscan, 2), "\n")


# WIZUALIZACJE
# Porównanie klastrów w różnych algorytmach
fviz_cluster(list(data = powers_sample, cluster = powers_sample$cluster_kmeans),
             geom = "point",
             palette = "jco", 
             ellipse.type = "t",
             main = "K-means Clustering")
fviz_cluster(pam.res, 
             geom = "point", 
             palette = "jco", 
             ellipse.type = "t",
             main = "PAM Clustering")
fviz_cluster(powers_sample.dbscanBEST, data = powers_sample, geom = "point", ellipse.type = "t") +
  ggtitle(paste("DBSCAN eps =", eps, ", minPts =", minPts))

# NAJLEPSZE WYNIKI DAJE DBSCAN.


# Porównanie wyników grupowania do faktycznych grup w danych Pokmony: Typ i Generacja:
length(pokemon$Cluster_kmeans)
length(pokemon$Type.1)
length(pokemon$Generation)
nrow(pokemon)

# 1. K-MEANS
# Porównanie grup k-means z Type.1
table(pokemon$Cluster_kmeans, pokemon$Type.1)
# Porównanie grup k-means z Generation
table(pokemon$Cluster_kmeans, pokemon$Generation)

# Wizualizacja wyników K-MEANS vs. TYPE.1:
type_counts <- as.data.frame(table(pokemon$Cluster_kmeans, pokemon$Type.1))
colnames(type_counts) <- c("Cluster", "Type", "Count")

ggplot(type_counts, aes(x = Type, y = Count, fill = as.factor(Cluster))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie liczby Pokemonów według typu i grup k-means",
       x = "Typ Pokemona",
       y = "Liczba Pokemonów",
       fill = "Grupa k-means") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Wizualizacja wyników K-MEANS vs. GENERATION:
generation_counts <- as.data.frame(table(pokemon$Cluster_kmeans, pokemon$Generation))
colnames(generation_counts) <- c("Cluster", "Generation", "Count")

ggplot(generation_counts, aes(x = Generation, y = Count, fill = as.factor(Cluster))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie liczby Pokemonów według generacji i grup k-means",
       x = "Generacja",
       y = "Liczba Pokemonów",
       fill = "Grupa k-means")


# 2. DBSCAN
# Porównanie grup DBSCAN z Type.1
table(pokemon$Cluster_DBSCAN, pokemon$Type.1)
# Porównanie grup DBSCAN z Generation
table(pokemon$Cluster_DBSCAN, pokemon$Generation)


# Wizualizacja wyników DBSCAN vs Type.1
type_counts_dbscan <- as.data.frame(table(pokemon$Cluster_DBSCAN, pokemon$Type.1))
colnames(type_counts_dbscan) <- c("Cluster", "Type", "Count")

ggplot(type_counts_dbscan, aes(x = Type, y = Count, fill = as.factor(Cluster))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie liczby Pokemonów według typu i grup DBSCAN",
       x = "Typ Pokemona",
       y = "Liczba Pokemonów",
       fill = "Grupa DBSCAN") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Wizualizacja wyników DBSCAN vs Generation
generation_counts_dbscan <- as.data.frame(table(pokemon$Cluster_DBSCAN, pokemon$Generation))
colnames(generation_counts_dbscan) <- c("Cluster", "Generation", "Count")

ggplot(generation_counts_dbscan, aes(x = Generation, y = Count, fill = as.factor(Cluster))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie liczby Pokemonów według generacji i grup DBSCAN",
       x = "Generacja",
       y = "Liczba Pokemonów",
       fill = "Grupa DBSCAN")


# Wnioski:
# Większość Pokemonów w DBSCAN, podobnie jak w przypadku k-means, trafia do jednej z grup.
# Jest też odsetek pokemontów, które trafiają do klastra 0, czyli grupy "szumów", nie pasują do żadnego z klastrów.

# Klaster 1 wydaje się bardziej zróżnicowany pod względem typów, chociaż dominują w nim Pokémony o popularnych typach jak Normal, Water i Flying.
# Klaster 2 ma wyraźną dominację w typach Water, Normal i Flying, ale jest bardziej jednolity, szczególnie w generacjach, gdzie występuje większa liczba Pokémonów z generacji 1, 3 i 5.
# Zauważalne jest, że oba klastry zawierają Pokémony z różnych generacji, ale Klaster 1 ma bardziej równomiernie rozłożoną liczbę Pokémonów między generacjami, podczas gdy Klaster 2 jest bardziej skoncentrowany w generacjach 1, 3 i 5.


# Statystyki dla Pokémonów w klastrze 1
summary(pokemon[pokemon$Cluster_DBSCAN == 1, c("HP", "Attack", "Defense", "Speed")])
# Statystyki dla Pokémonów w klastrze 2
summary(pokemon[pokemon$Cluster_DBSCAN == 2, c("HP", "Attack", "Defense", "Speed")])


# Test chi-kwadrat dla typów i generacji Pokémonów w klastrach DBSCAN:
type_cluster_table <- table(pokemon$Cluster_DBSCAN, pokemon$Type.1)
chi_square_type <- chisq.test(type_cluster_table)
chi_square_type

generation_cluster_table <- table(pokemon$Cluster_DBSCAN, pokemon$Generation)
chi_square_generation <- chisq.test(generation_cluster_table)
chi_square_generation

# Wyniki testów chi-kwadrat wskazują, że zarówno dla typów Pokémonów, jak i generacji Pokémonów, 
# nie ma statystycznie istotnej zależności między tymi zmiennymi a przynależnością do klastrów DBSCAN:


# Obliczanie średnich dla cech w klastrach
pokemon_cluster_means <- pokemon %>%
  group_by(Cluster_DBSCAN) %>%
  summarise(
    avg_hp = mean(HP, na.rm = TRUE),
    avg_attack = mean(Attack, na.rm = TRUE),
    avg_defense = mean(Defense, na.rm = TRUE),
    avg_speed = mean(Speed, na.rm = TRUE)
  )
pokemon_cluster_means


# Test ANOVA dla HP
anova_hp <- aov(HP ~ Cluster_DBSCAN, data = pokemon)
summary(anova_hp)
# Test ANOVA dla Attack
anova_attack <- aov(Attack ~ Cluster_DBSCAN, data = pokemon)
summary(anova_attack)
# Test ANOVA dla Defense
anova_defense <- aov(Defense ~ Cluster_DBSCAN, data = pokemon)
summary(anova_defense)
# Test ANOVA dla Speed
anova_speed <- aov(Speed ~ Cluster_DBSCAN, data = pokemon)
summary(anova_speed)


# Wyniki testów ANOVA wskazują, że dla żadnej z cech (HP, Attack, Defense, Speed) nie znaleziono statystycznie
# istotnych różnic między klastrami DBSCAN (wszystkie p-value > 0.05).
# Oznacza to, że różnice w średnich wartościach tych cech w klastrach nie są statystycznie istotne, co sugeruje,
# że klasteryzacja DBSCAN nie rozdziela Pokémonów na grupy o wyraźnie różnych wartościach tych cech.