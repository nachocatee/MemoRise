import { View, Image, StyleSheet, Pressable } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import {
  calculateDynamicHeight,
  calculateDynamicWidth,
} from "../../constants/dynamicSize";

const LandingScreen = () => {
  return (
    <LinearGradient
      //   colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <View style={styles.rootContainer}>
        <View style={{ alignItems: "center" }}>
          <Image
            source={require("../../assets/image/note_sm.png")}
            style={imageStyle}
          />
          <Image
            source={require("../../assets/image/logo/logo.png")}
            style={[logoStyle, styles.logo]}
          />
          <Pressable style={styles.kakao}>
            <Image
              source={require("../../assets/image/kakao.png")}
              style={kakaoStyle}
            />
          </Pressable>
        </View>
      </View>
    </LinearGradient>
  );
};

export default LandingScreen;

const styles = StyleSheet.create({
  rootContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  logo: {
    position: "absolute",
    top: "27%",
  },
  kakao: {
    position: "absolute",
    bottom: "27%",
  },
});

const imageStyle = {
  width: calculateDynamicWidth(320),
  height: calculateDynamicHeight(320, 333, calculateDynamicWidth(320)),
};

const logoStyle = {
  width: calculateDynamicWidth(253),
  height: calculateDynamicHeight(253, 48, calculateDynamicWidth(253)),
};

const kakaoStyle = {
  width: calculateDynamicWidth(230),
  height: calculateDynamicHeight(230, 48, calculateDynamicWidth(230)),
};
