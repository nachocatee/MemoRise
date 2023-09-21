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
    private String profile;

    @Enumerated(EnumType.STRING)
    private Role role;

    private int isDeleted;
    private LocalDateTime deletedAt;

    @Builder
    public User(String email, String nickname, String profile, Role role) {
        this.email = email;
        this.nickname = nickname;
        this.profile = profile;
        this.role = role;
    }

    public void update(String nickname, String profile) {
        this.nickname = nickname;
        this.profile = profile;
    }

    public void delete() {
        this.isDeleted = 1;
        this.deletedAt = LocalDateTime.now();
    }

    public String getRoleKey() {
        return this.role.getKey();
    }

}
