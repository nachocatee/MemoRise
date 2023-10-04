import React from "react";
import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import {
  View,
  StyleSheet,
  Image,
  Pressable,
  GestureResponderEvent,
} from "react-native";

type RootStackParamList = {
  Menu: undefined;
  Main: undefined;
};

type HeaderProps = {
  openModal: (event: GestureResponderEvent) => void;
};

const MainHeader: React.FC<HeaderProps> = ({ openModal }) => {
  const navigation = useNavigation<StackNavigationProp<RootStackParamList>>();

  return (
    <View style={styles.header}>
      <Pressable style={styles.notificationContainer} onPress={openModal}>
        <View style={styles.padding}>
          <Image
            source={require("../../assets/icons/help.png")}
            style={styles.notification}
          />
        </View>
      </Pressable>
      <Pressable onPress={() => navigation.navigate("Main")}>
        <Image
          source={require("../../assets/image/logo/logowhite.png")}
          style={styles.logo}
        />
      </Pressable>
      <Pressable
        style={styles.menuContainer}
        onPress={() => navigation.navigate("Menu")}
      >
        <View style={styles.padding}>
          <View style={styles.menu}></View>
        </View>
      </Pressable>
    </View>
  );
};

export default MainHeader;

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
  notificationContainer: {
    position: "absolute",
    left: 0,
  },
  notification: {
    width: 30,
    height: 30,
  },
  menuContainer: {
    position: "absolute",
    right: 0,
  },
  menu: {
    width: 31,
    height: 15,
    borderTopWidth: 2,
    borderTopColor: "white",
    borderBottomWidth: 2,
    borderBottomColor: "white",
  },
  padding: {
    padding: 10,
  },
});
