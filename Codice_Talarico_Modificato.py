from os import system
import matplotlib.pyplot as plt
import numpy as np 

system("cls")
#creo la funzione sigmoide che mi servirà per i calcoli
def sigmoide(previsione: float):
    return 1 / (1 + np.exp(- previsione))

#creo la funzione derivata parziale che mi servirà per i calcoli
def derivata_parziale(previsione: float, obiettivo: int, peso: float):
    return (2 * (previsione - obiettivo)) * (previsione * (1 - previsione)) * peso

#creo la funzione percentale che mi servirà per i calcoli
def percentuale(affidabilità: float , data_set_prova):
    # affidabilità è il return della funzione risultati
    return affidabilità / len(data_set_prova)* 100

#creo la funzione sigmoide che mi servirà per i calcoli
def risultati(data_set_allenamento , data_set_prova ):
    #alleno i pesi con la funzione allenamento
    peso_1 , peso_2 , bias = allenamento(data_set_allenamento)
    #affidabiltà sarà la nostra variabbile contatore dove andrà ad incrementarsi ogni volta che la previsone sarà uguale al nostro obiettivo
    affidabilità = 0
    for dato in data_set_prova:
        previsione = sigmoide(dato[0] * peso_1 + dato[1] * peso_2 + bias)  
        if abs(round(previsione)) == dato[2]:
            affidabilità += 1

    return affidabilità

#creo la funzione allenamento che ci servirà per allenare i nostri pesi
def allenamento(data_set_allenamento):
    #questa riga ci serve per far in modo che il random dei pesi verrà randomizzata una sola votra quando richiamiamo la funzione
    np.random.seed(1)

    peso_1, peso_2, bias = np.random.random(), np.random.random(), np.random.random()
    learning_rate = 0.1
    
    for epoca in range(10_000):
        #utilizzando la funzione random.randit possiamo prendere un idice casuale da dentro la lista data_set_allenan
        indice_casuale = np.random.randint(0, len(data_set_allenamento) - 1)
        dato_random = data_set_allenamento[indice_casuale]
        
        neurone = dato_random[0] * peso_1 + dato_random[1] * peso_2 + bias
        
        previsione = sigmoide(neurone)
        
        obiettivo = dato_random[2]
        
        peso_1 -= learning_rate * derivata_parziale(previsione, obiettivo, dato_random[0])
        peso_2 -= learning_rate * derivata_parziale(previsione, obiettivo, dato_random[1])
        bias -= learning_rate * derivata_parziale(previsione, obiettivo, 1)
                    
    return peso_1, peso_2, bias

#creo una funzione per creare il grafico
def grafico(nomi_test , percentuali):
    #con questa riga do le 2 dimensioni della finesta
    plt.figure(figsize = (15 , 7))
    #con questa riga do gli input che mi serviranno dentro al grafico , in piu do anche i colori per le barre(colore della barra e del bordo)
    plt.bar(nomi_test , percentuali , color = "#ABCDEF" , edgecolor = "black")
    #con questa riga do un nome per l'asse y
    plt.ylabel("Percentuali")
    #con questa riga do un titolo alla finesta
    plt.title("Rete Neurale")
    #renderizzo il tutto
    plt.show()


def main() -> None:
    
    data_set_allenamento = [
        [9, 7.0, 0],
        [2, 5.0, 1],
        [3.2, 4.94, 1],
        [9.1, 7.46, 0],
        [1.6, 4.83, 1],
        [8.4, 7.46, 0],
        [8, 7.28, 0],
        [3.1, 4.58, 1],
        [6.3, 9.14, 0],
        [3.4, 5.36, 1]
    ] 
    
    data_set_mio = [
        [8, 5.0, 0],
        [1, 4.0, 1],
        [2.2, 3.94, 1],
        [8.1, 6.46, 0],
        [1.4, 4.80, 1],
        [7.4, 7.40, 0],
        [7, 7.10, 0],
        [2.1, 3.58, 1],
        [5.3, 8.14, 0],
        [2.4, 4.36, 1]
    ] 

    test_1 = (percentuale(risultati(data_set_allenamento , data_set_allenamento ) , data_set_allenamento))
    test_2 = (percentuale(risultati(data_set_allenamento[:5] , data_set_allenamento[5:] ) , data_set_allenamento[5:]))
    test_3 = (percentuale(risultati(data_set_mio , data_set_mio ) , data_set_mio))
    test_4 = (percentuale(risultati(data_set_mio[:5] , data_set_mio[5:] ) , data_set_mio[5:]))
    test_5 = (percentuale(risultati(data_set_allenamento , data_set_mio ) , data_set_mio))

    
    nomi_test = ["Dataset Talarico" , "Dataset Talarico[:5-5:]", "Dataset Mio" , "DaTaset Mio[:5-5:]" , "Dataset Talarico[:5} Dataset Mio[5:]"]
    percentuali = [test_1 , test_2 , test_3 , test_4 , test_5]

    grafico(nomi_test , percentuali)

main()
