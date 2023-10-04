import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import { View, StyleSheet, Image, Pressable } from "react-native";
import Colors from "../../constants/colors";

type RootStackParamList = {
  Main: undefined;
  Menu: undefined;
};

const GoBackHeader = () => {
  const navigation =
    useNavigation<StackNavigationProp<RootStackParamList, "Main">>();

  return (
    <View style={styles.header}>
      <Pressable
        onPress={() => navigation.goBack()}
        style={styles.goBackContainer}
      >
        <View style={styles.padding}>
          <Image
            source={require("../../assets/icons/goback.png")}
            style={styles.goBack}
          />
        </View>
      </Pressable>
      <Pressable onPress={() => navigation.replace("Main")}>
        <Image
          source={require("../../assets/image/logo/logoblack.png")}
          style={styles.logo}
        />
      </Pressable>
      <Pressable
        onPress={() => navigation.navigate("Menu")}
        style={styles.menuContainer}
      >
        <View style={styles.padding}>
          <View style={styles.menu}></View>
        </View>
      </Pressable>
    </View>
  );
};

export default GoBackHeader;

const styles = StyleSheet.create({
  header: {
    height: 97,
    justifyContent: "center",
    alignItems: "center",
    flexDirection: "row",
    marginHorizontal: 15,
  },
  logo: {
    width: 189,
    height: 35,
  },
  goBackContainer: {
    position: "absolute",
    left: 0,
  },
  goBack: {
    width: 13,
    height: 25,
  },
  menuContainer: {
    position: "absolute",
    right: 0,
  },
  menu: {
    width: 31,
    height: 15,
    borderTopWidth: 2,
    borderTopColor: Colors.text,
    borderBottomWidth: 2,
    borderBottomColor: Colors.text,
  },
  padding: {
    padding: 10,
  },
});
