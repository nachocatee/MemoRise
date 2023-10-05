import React from "react";
import { View, Text, StyleSheet, Image, Pressable } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import Colors from "../constants/colors";
import { calculateDynamicWidth } from "../constants/dynamicSize";

type GroupBoxProps = {
  teamName: string;
  myProfile: string;
  memberProfiles: string[];
  ownerProfile: string | null;
  owner: boolean;
  teamSeq: number;
  goDetailHandler: (teamSeq: number) => void;
};

const GroupBox: React.FC<GroupBoxProps> = ({
  teamName,
  myProfile,
  memberProfiles,
  ownerProfile,
  owner,
  teamSeq,
  goDetailHandler,
}) => {
  return (
    <Pressable
      style={{ marginHorizontal: calculateDynamicWidth(8) }}
      onPress={() => goDetailHandler(teamSeq)}
    >
      <LinearGradient
        colors={["#F5F5F5", Colors.hover]}
        start={{ x: 0, y: 0 }}
        end={{ x: 0, y: 1 }}
        style={styles.groupBox}
      >
        <View>
          <Image style={styles.profileImg} source={{ uri: myProfile }} />
          {owner && (
            <Image
              style={styles.crown}
              source={require("../assets/image/crown.png")}
            />
          )}
        </View>
        {ownerProfile && (
          <View>
            <Image style={styles.profileImg} source={{ uri: ownerProfile }} />
          </View>
        )}
        {memberProfiles.map((member, idx) => (
          <View key={idx}>
            {member !== myProfile && (
              <Image style={styles.profileImg} source={{ uri: member }} />
            )}
          </View>
        ))}
      </LinearGradient>
      <Text style={styles.title} numberOfLines={1} ellipsizeMode="tail">
        {teamName}
      </Text>
    </Pressable>
  );
};

export default GroupBox;

const styles = StyleSheet.create({
  groupBox: {
    width: calculateDynamicWidth(128),
    height: calculateDynamicWidth(128),
    borderRadius: calculateDynamicWidth(15),
    flexDirection: "row",
    flexWrap: "wrap",
  },
  title: {
    color: Colors.text,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(18),
    textAlign: "center",
    marginTop: calculateDynamicWidth(5),
    marginBottom: calculateDynamicWidth(30),
    width: calculateDynamicWidth(128),
  },
  profileImg: {
    width: calculateDynamicWidth(60),
    height: calculateDynamicWidth(60),
    borderRadius: calculateDynamicWidth(15),
    margin: calculateDynamicWidth(2),
  },
  crown: {
    width: calculateDynamicWidth(21.86),
    height: calculateDynamicWidth(18),
    position: "absolute",
    left: "50%",
    transform: [{ translateX: -calculateDynamicWidth(21.86) / 2 }],
    top: -calculateDynamicWidth(12),
  },
});
