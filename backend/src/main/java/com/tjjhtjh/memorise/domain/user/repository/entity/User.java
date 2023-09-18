package com.tjjhtjh.memorise.domain.user.repository.entity;

import com.tjjhtjh.memorise.global.entity.BaseEntity;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class User extends BaseEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long userSeq;

    private String email;
    private String nickname;
    private String password;
    private String profileImg;

    @Enumerated(EnumType.STRING)
    private Role role;

    private int isDeleted;
    private LocalDateTime deletedAt;

    @Builder
    public User(Long userSeq, String email, String nickname, String profileImg) {
        this.userSeq = userSeq;
        this.email = email;
        this.nickname = nickname;
        this.profileImg = profileImg;
    }

    public void update(String nickname, String profileImg) {
        this.nickname = nickname;
        this.profileImg = profileImg;
    }

    public void delete() {
        this.isDeleted = 1;
        this.deletedAt = LocalDateTime.now();
    }

}
